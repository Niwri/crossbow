import time

import click
import cv2 as cv
import numpy as np
from serial import Serial
import random 

PORT1 = "COM4"
PORT2 = "COM5"
BAUDRATE = 115200

PREAMBLE = "!START!\r\n"
DELTA_PREAMBLE = "!DELTA!\r\n"
SUFFIX = "!END!\r\n"

ROWS = 144
COLS = 174

def monitor(
    port: str,
    baudrate: int,
    timeout: int,
    rows: int,
    cols: int,
    preamble: str,
    delta_preamble: str,
    suffix: str,
    short_input: bool,
    rle: bool,
    quiet: bool,
    ser: Serial
) -> None:
    """
    Display images transferred through serial port. Press 'q' to close.
    """
    prev_frame_ts = None  # keep track of frames per second
    frame = None

    click.echo(f"Opening communication on port {port} with baudrate {baudrate}")

    if isinstance(suffix, str):
        suffix = suffix.encode("ascii")

    if isinstance(preamble, str):
        preamble = preamble.encode("ascii")

    if isinstance(delta_preamble, str):
        delta_preamble = delta_preamble.encode("ascii")

    img_rx_size = rows * cols
    if short_input:
        img_rx_size //= 2
    if rle:
        img_rx_size *= 2

    partial_frame_counter = 0  # count partial updates every full frame

    while True:
        if not quiet:
            click.echo("Waiting for input data...")

        full_update = wait_for_preamble(ser, preamble, delta_preamble)

        if full_update:
            click.echo(
                f"Full update (after {partial_frame_counter} partial updates)"
            )
            partial_frame_counter = 0
        else:
            if not quiet:
                click.echo("Partial update")
            partial_frame_counter += 1

            if frame is None:
                click.echo(
                    "No full frame has been received yet. Skipping partial update."
                )
                continue

        if not quiet:
            click.echo("Receiving picture...")

        try:
            raw_data = get_raw_data(ser, img_rx_size, suffix)
            if not quiet:
                click.echo(f"Received {len(raw_data)} bytes")
        except ValueError as e:
            click.echo(f"Error while waiting for frame data: {e}")

        if short_input:
            raw_data = (
                expand_4b_to_8b(raw_data)
                if not rle
                else expand_4b_to_8b_rle(raw_data)
            )
        elif rle and len(raw_data) % 2 != 0:
            # sometimes there serial port picks up leading 0s
            # discard these
            raw_data = raw_data[1:]

        if rle:
            raw_data = decode_rle(raw_data)

        try:
            new_frame = load_raw_frame(raw_data, rows, cols)

        except ValueError as e:
            click.echo(f"Malformed frame. {e}")
            continue

        frame = new_frame if full_update else frame + new_frame
        return frame


def wait_for_preamble(ser: Serial, preamble: str, partial_preamble: str) -> bool:
    """
    Wait for a preamble string in the serial port.

    Returns `True` if next frame is full, `False` if it's a partial update.
    """
    while True:
        try:
            line = ser.readline()
            if line == preamble:
                return True
            elif line == partial_preamble:
                return False
        except UnicodeDecodeError:
            pass


def get_raw_data(ser: Serial, num_bytes: int, suffix: bytes = b"") -> bytes:
    """
    Get raw frame data from the serial port.
    """
    rx_max_len = num_bytes + len(suffix)
    max_tries = 10_000
    raw_img = b""

    for _ in range(max_tries):
        raw_img += ser.read(max(1, ser.in_waiting))

        suffix_idx = raw_img.find(suffix)
        if suffix_idx != -1:
            raw_img = raw_img[:suffix_idx]
            break

        if len(raw_img) >= rx_max_len:
            raw_img = raw_img[:num_bytes]
            break
    else:
        raise ValueError("Max tries exceeded.")

    return raw_img


def expand_4b_to_8b(raw_data: bytes) -> bytes:
    """
    Transform an input of 4-bit encoded values into a string of 8-bit values.

    For example, value 0xFA gets converted to [0xF0, 0xA0]
    """
    return bytes(val for pix in raw_data for val in [pix & 0xF0, (pix & 0x0F) << 4])


def expand_4b_to_8b_rle(raw_data: bytes) -> bytes:
    """
    Transform an input of 4-bit encoded RLE values into a string of 8-bit values.

    For example, value 0xFA gets converted to [0xF0, 0x0A]
    """
    return bytes(val for pix in raw_data for val in [pix & 0xF0, pix & 0x0F])


def decode_rle(raw_data: bytes) -> bytes:
    """
    Decode Run-Length Encoded data.
    """
    return bytes(
        val
        for pix, count in zip(raw_data[::2], raw_data[1::2])
        for val in [pix] * count
    )


def load_raw_frame(raw_data: bytes, rows: int, cols: int) -> np.array:

    print("Raw Data in Binary:")
    count = 0
    data = np.zeros((int(rows/2), int(cols), 1), np.uint16)

    for i in range(int(rows/2)):
        for j in range(int(cols)):
            index = int((i * cols + j) * 2)
            byte1 = raw_data[index+1]
            if index+2 == rows*cols:
                byte2 = 0x0
            else:
                byte2 = raw_data[index+2]
            data[i][j] = byte1 << 8 | byte2
           
            if count < 100:
                count = count + 1
                print(bin(byte1 << 8 | byte2)[2:].zfill(16), end=' ')
    print()  # Add a newline after printing all bytes

   

    #data = np.frombuffer(raw_data, dtype=np.uint16).reshape((int(rows), int(cols/2), 1))
    ycrcb_frame = np.zeros((int(rows/2), int(cols), 3), np.uint8)
    count = 0
    y_0 = 0
    y_1 = 0
    cb = 0
    cr = 0
    for i in range(int(rows/2)):
        for j in range(int(cols)):
            """
            rgb_frame[i][j][2] = (data[i][j][0] & 0b1111100000000000) >> 8
            rgb_frame[i][j][1] = (data[i][j][0] & 0b0000011111100000) >> 3
            rgb_frame[i][j][0] = (data[i][j][0] & 0b0000000000011111) << 3
            if count <= 100:
                count = count + 1
                print(bin(data[i][j][0])[2:].zfill(16), end=' ')
                print(data[i][j], ycrcb_frame[i][j])
    
            """
            if j % 2 == 0:
                y_0 = (data[i][j][0] & 0xFF00) >> 8
                cb = (data[i][j][0] & 0xFF)
            else:
                y_1 = (data[i][j][0] & 0xFF00) >> 8 
                cr = (data[i][j][0] & 0xFF)
                ycrcb_frame[i][j-1][0] = y_0
                ycrcb_frame[i][j-1][1] = cr
                ycrcb_frame[i][j-1][2] = cb
                ycrcb_frame[i][j][0] = y_1
                ycrcb_frame[i][j][1] = cr
                ycrcb_frame[i][j][2] = cb
    rgb_frame = cv.cvtColor(ycrcb_frame, cv.COLOR_YCR_CB2BGR)
    return rgb_frame


@click.command()
@click.option(
    "-p1", "--port1", default=PORT1, help="Serial (COM) port of the target board"
)
@click.option(
    "-p2", "--port2", default=PORT2, help="Serial (COM) port of the target board"
)
@click.option("-br", "--baudrate", default=BAUDRATE, help="Serial port baudrate")
@click.option("--timeout", default=1, help="Serial port timeout")
@click.option("--rows", default=ROWS, help="Number of rows in the image")
@click.option("--cols", default=COLS, help="Number of columns in the image")
@click.option("--preamble", default=PREAMBLE, help="Preamble string before the frame")
@click.option(
    "--delta_preamble",
    default=DELTA_PREAMBLE,
    help="Preamble before a delta update during video compression.",
)
@click.option(
    "--suffix", default=SUFFIX, help="Suffix string after receiving the frame"
)
@click.option(
    "--short-input",
    is_flag=True,
    default=False,
    help="If true, input is a stream of 4b values",
)
@click.option("--rle", is_flag=True, default=False, help="Run-Length Encoding")
@click.option("--quiet", is_flag=True, default=False, help="Print fewer messages")
def main(port1: str, port2: str, 
        baudrate: int,
        timeout: int,
        rows: int,
        cols: int,
        preamble: str,
        delta_preamble: str,
        suffix: str,
        short_input: bool,
        rle: bool,
        quiet: bool,
    ):
    while True:
        rgb_arr1 = []
        rgb_arr2 = []
        input("Click a button to capture Image One")
        with Serial(port1, BAUDRATE, timeout=1) as ser:
            ser.write(b'\r\n!BEGIN!\r\n')
            rgb_arr1 = monitor(
                            port1,
                            baudrate,
                            timeout,
                            rows,
                            cols,
                            preamble,
                            delta_preamble,
                            suffix,
                            short_input,
                            rle,
                            quiet,
                            ser
                        )
        
        print(rgb_arr1)
        cv.namedWindow("Video Stream1", cv.WINDOW_KEEPRATIO)
        cv.imshow("Video Stream1", rgb_arr1)
        cv.waitKey(1)

        input("Click a button to capture Image Two")
        with Serial(port2, BAUDRATE, timeout=1) as ser:
            ser.write(b'\r\n!BEGIN!\r\n')
            rgb_arr2 = monitor(
                        port2,
                        baudrate,
                        timeout,
                        rows,
                        cols,
                        preamble,
                        delta_preamble,
                        suffix,
                        short_input,
                        rle,
                        quiet,
                        ser
                    )

        cv.namedWindow("Video Stream2", cv.WINDOW_KEEPRATIO)
        cv.imshow("Video Stream2", rgb_arr2)
        cv.waitKey(1)

if __name__ == "__main__":
    main()
    #monitor()
    """
    rows = 400
    cols = 400
    sum = np.zeros((rows,cols,3), np.uint8) 

    while(1):
        
        for i in range(rows):
            for j in range(cols):
                sum[i][j] = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
        cv.imshow("Video Stream", sum)
        if cv.waitKey(1) == ord("q"):
                break
    """
    """
    while True:
        a = b""
        with Serial("COM4", BAUDRATE, timeout=1) as ser:
            ser.write(b'\r\n!BEGIN!\r\n')
            print("Sent\n")
            while True:
                a += ser.read(max(1, ser.in_waiting))
                suffix_idx = a.find("WACK\n".encode("ascii"))
                if suffix_idx != -1:
                    break
            print(a)
            print("END")
    """