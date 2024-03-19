/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2023 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <string.h>
#include <stdio.h>
#include "config.h"
#include "ov7670.h"
#include "stdlib.h"

/* USER CODE BEGIN PV */
#define PREAMBLE "\r\n!START!\r\n"
#define DELTA_PREAMBLE "\r\n!DELTA!\r\n"
#define SUFFIX "!END!\r\n"

uint16_t snapshot_buff[IMG_ROWS * IMG_COLS];
//uint8_t old_snapshot_buff[IMG_ROWS * IMG_COLS];

uint8_t tx_buff[sizeof(PREAMBLE) + 2 * IMG_ROWS * IMG_COLS + sizeof(SUFFIX)];
uint8_t previous_buff[IMG_ROWS*IMG_COLS];
size_t tx_buff_len = 0;

uint8_t dma_flag = 0, dma2_flag = 0;

// Your function definitions here
void print_buf(void);
void reset_buffer(int len);

void print_buf_delta(void);
void print_buf_RLE(void);
void print_buf_RGB(void);

int main(void)
{
  /* Reset of all peripherals */
  HAL_Init();

  /* Configure the system clock */
  SystemClock_Config();

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_DMA_Init();
  MX_DCMI_Init();
  MX_USART3_UART_Init();
  MX_USB_OTG_FS_PCD_Init();
  MX_I2C2_Init();
  MX_TIM1_Init();
  MX_TIM6_Init();

  char msg[100];

  HAL_TIM_PWM_Start(&htim1, TIM_CHANNEL_1);
  ov7670_init();
  HAL_Delay(100);
	uint8_t count = 0;
	/*
	while(1) {
		ov7670_capture(snapshot_buff);
		//print_buf_RLE(); // Section 6.3
		//print_buf(); // Section 4 or 6.2
		// Section 7
		if(count % 5 == 0) {
			print_buf_RLE();
		} else {
			print_buf_delta();
		}
		count++;

		
		print_buf_RGB();
	}
	*/
  // Your startup code here
	// 2.1 Test
	/*memcpy(tx_buff, PREAMBLE, sizeof(PREAMBLE));
	for(int i = 0; i < 25056; i++){
		*(tx_buff+sizeof(PREAMBLE)+i) = 0xFF;
	}
	memcpy(tx_buff + sizeof(PREAMBLE) + 2 * IMG_ROWS * IMG_COLS, SUFFIX, sizeof(SUFFIX));
	tx_buff_len = sizeof(PREAMBLE) + 2 * IMG_ROWS * IMG_COLS + sizeof(SUFFIX);
	uart_send_bin(tx_buff, tx_buff_len); */
	
  while (1)
  {
    // Your code here
    if (HAL_GPIO_ReadPin(USER_Btn_GPIO_Port, USER_Btn_Pin)) {
      HAL_Delay(100);  // debounce
			ov7670_capture(snapshot_buff);
			HAL_Delay(0);
			
			print_buf_RGB();
    }
  }
}


void print_buf() {
  // Send image data through serial port.
  // Your code here
	while(HAL_UART_GetState(&huart3) != HAL_UART_STATE_READY){
		HAL_Delay(0);
	}
	// Part 4
	
	memcpy(tx_buff, PREAMBLE, sizeof(PREAMBLE));
	for(int i = 0; i < (IMG_COLS*IMG_ROWS); i++){
		*(tx_buff+sizeof(PREAMBLE)+i) = (uint8_t)((*(snapshot_buff+i) & 0xFF00) >> 8);
	}
	memcpy(tx_buff + sizeof(PREAMBLE) + 2*(IMG_ROWS * IMG_COLS), SUFFIX, sizeof(SUFFIX));
	tx_buff_len = sizeof(PREAMBLE) + 2*(IMG_ROWS * IMG_COLS) + sizeof(SUFFIX);
	
	
	// Part 6.2
	/*
	memcpy(tx_buff, PREAMBLE, sizeof(PREAMBLE));
	for(int i = 0; i < (IMG_COLS*IMG_ROWS)/2; i++){
		*(tx_buff+sizeof(PREAMBLE)+i) = (uint8_t)(((*(snapshot_buff+(2*i)) & 0xF000) >> 8) | ((*(snapshot_buff+(2*i)+1) & 0xF000) >> 12));
	}
	memcpy(tx_buff + sizeof(PREAMBLE) + (IMG_ROWS * IMG_COLS), SUFFIX, sizeof(SUFFIX));
	tx_buff_len = sizeof(PREAMBLE) + (IMG_ROWS * IMG_COLS) + sizeof(SUFFIX);
	*/
	uart_send_bin(tx_buff, tx_buff_len);
	HAL_DCMI_Resume(&hdcmi);
}

void reset_buffer(int len) {
	for(int i = 0; i < tx_buff_len; i++)
		*(tx_buff+i) = 0;
	
}

void print_buf_delta() {
	// Send image data through serial port.
  // Your code here
	while(HAL_UART_GetState(&huart3) != HAL_UART_STATE_READY){
		HAL_Delay(0);
	}
	
	// Part 6.3
	uint16_t count = 0;
	uint8_t start = 1;
	int8_t previous_delta = 0x00;
	int8_t current_delta = 0x00;
	uint8_t current = 0x00;
	int8_t delta = 0x00;
	uint8_t pixel_count = 0x1;
	int total_count = 1;
	reset_buffer(tx_buff_len);
	memcpy(tx_buff, DELTA_PREAMBLE, sizeof(DELTA_PREAMBLE));
	for(int i = 0; i < (IMG_COLS*IMG_ROWS); i++){
		current = (uint8_t)((*(snapshot_buff+i) & 0xF000) >> 8);
		current_delta = current - previous_buff[i];
		if(start == 1) { // If first pixel, use default values
			start = 0;
		} else if(previous_delta == current_delta) { // If same pixel as previous, increment counter
			pixel_count += 1;
			total_count +=1;
		} else { // If different pixel from previous, move to next byte and reset counter
			count += 1;
			pixel_count = 1;
			total_count += 1;
		}
		
		if(pixel_count > 0xF) { // If the pixel counter exceeds 4 bit limit
			count += 1;
			pixel_count = 1;
		}
	
		previous_delta = current_delta;
		previous_buff[i] = current;
		*(tx_buff+sizeof(DELTA_PREAMBLE)+count) = current_delta | pixel_count;
		
		
	}
		
	memcpy(tx_buff + sizeof(DELTA_PREAMBLE) + 2*(count+1), SUFFIX, sizeof(SUFFIX));
	tx_buff_len = sizeof(DELTA_PREAMBLE) + 2*(count+1) + sizeof(SUFFIX);
	
	uart_send_bin(tx_buff, tx_buff_len);
	HAL_DCMI_Resume(&hdcmi);
	
}


void print_buf_RLE() {
	// Send image data through serial port.
  // Your code here
	while(HAL_UART_GetState(&huart3) != HAL_UART_STATE_READY){
		HAL_Delay(0);
	}
	
	// Part 6.3
	uint16_t count = 0;
	uint8_t start = 1;
	uint8_t previous = 0x00;
	uint8_t current = 0x00;
	uint8_t pixel_count = 0x1;
	int total_count = 1;
	reset_buffer(tx_buff_len);
	memcpy(tx_buff, PREAMBLE, sizeof(PREAMBLE));
	for(int i = 0; i < (IMG_COLS*IMG_ROWS); i++){
		current = (uint8_t)((*(snapshot_buff+i) & 0xF000) >> 8);
		if(start == 1) { // If first pixel, use default values
			start = 0;
		} else if(previous == current) { // If same pixel as previous, increment counter
			pixel_count += 1;
			total_count +=1;
		} else { // If different pixel from previous, move to next byte and reset counter
			count += 1;
			pixel_count = 1;
			total_count += 1;
		}
		
		if(pixel_count > 0xF) { // If the pixel counter exceeds 4 bit limit
			count += 1;
			pixel_count = 1;
		}
	
		previous = current;
		previous_buff[i] = current;
		*(tx_buff+sizeof(PREAMBLE)+count) = current | pixel_count;
		
		
	}
		
	memcpy(tx_buff + sizeof(PREAMBLE) + 2*(count+1), SUFFIX, sizeof(SUFFIX));
	tx_buff_len = sizeof(PREAMBLE) + 2*(count+1) + sizeof(SUFFIX);
	
	uart_send_bin(tx_buff, tx_buff_len);
	HAL_DCMI_Resume(&hdcmi);
	
}


void print_buf_RGB() {
  // Send image data through serial port.
  // Your code here
	while(HAL_UART_GetState(&huart3) != HAL_UART_STATE_READY){
		HAL_Delay(0);
	}
	// Part 4
	
	memcpy(tx_buff, PREAMBLE, sizeof(PREAMBLE));
	for(int i = 0; i < (IMG_COLS*IMG_ROWS)/2; i++){
		*(tx_buff+sizeof(PREAMBLE)+2*i) = (uint8_t)((*(snapshot_buff+i) & 0xFF00) >> 8);
		*(tx_buff+sizeof(PREAMBLE)+2*i+1) = (uint8_t)(*(snapshot_buff+i) & 0xFF);
	}
	memcpy(tx_buff + sizeof(PREAMBLE) + 2*(IMG_ROWS * IMG_COLS), SUFFIX, sizeof(SUFFIX));
	tx_buff_len = sizeof(PREAMBLE) + 2*(IMG_ROWS * IMG_COLS) + sizeof(SUFFIX);
	
	uart_send_bin(tx_buff, tx_buff_len);
	HAL_DCMI_Resume(&hdcmi);
}
