#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/i2c.h"
#include "esp_log.h"

#define I2C_MASTER_SCL_IO 22
#define I2C_MASTER_SDA_IO 21
#define I2C_MASTER_NUM I2C_NUM_0
#define MPU6050_ADDR 0x68

static const char *TAG = "HW_TEST";

void app_main(void) {
    // 1. Configure the I2C Master peripheral
    i2c_config_t conf = {
        .mode = I2C_MODE_MASTER,
        .sda_io_num = I2C_MASTER_SDA_IO,
        .scl_io_num = I2C_MASTER_SCL_IO,
        .sda_pullup_en = GPIO_PULLUP_ENABLE,
        .scl_pullup_en = GPIO_PULLUP_ENABLE,
        .master.clk_speed = 100000, 
    };
    i2c_param_config(I2C_MASTER_NUM, &conf);
    i2c_driver_install(I2C_MASTER_NUM, conf.mode, 0, 0, 0);

    // 2. Ping the WHO_AM_I register (Address 0x75)
    uint8_t reg_addr = 0x75;
    uint8_t data = 0;
    
    ESP_LOGI(TAG, "Testing I2C connection to MPU6050...");
    esp_err_t err = i2c_master_write_read_device(I2C_MASTER_NUM, MPU6050_ADDR, &reg_addr, 1, &data, 1, pdMS_TO_TICKS(1000));

    // 3. Evaluate the physical connection
    if (err == ESP_OK) {
        ESP_LOGI(TAG, "SUCCESS! MPU6050 WHO_AM_I register reads: 0x%02X", data);
        if (data == 0x68) {
            ESP_LOGI(TAG, "Sensor is genuine and I2C wiring is perfect.");
        } else {
            ESP_LOGW(TAG, "Connection made, but unexpected register value. Check module.");
        }
    } else {
        ESP_LOGE(TAG, "I2C Connection Failed. Check your 3.3V, GND, SDA, and SCL wires.");
    }
}