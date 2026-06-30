#ifndef AV1_ENCODER_LOG_PARTITION_H_
#define AV1_ENCODER_LOG_PARTITION_H_

#include <stdint.h>
#include <stdio.h>

#define AV1_PARTITION_LOG_MAX_BLOCK_SIZE 64

typedef struct {
  uint16_t frame_width;
  uint16_t frame_height;
  uint16_t blk_x;
  uint16_t blk_y;
  uint16_t blk_w;
  uint16_t blk_h;
  uint8_t partition_mode;
  uint8_t qindex;
  uint8_t level;
  uint8_t reserved;
  int16_t y_data[AV1_PARTITION_LOG_MAX_BLOCK_SIZE *
                 AV1_PARTITION_LOG_MAX_BLOCK_SIZE];
} av1_partition_sample_t;

#ifdef LOG_PARTITION_DATA
extern FILE *av1_partition_log_file;
void av1_partition_log_init(const char *filename);
void av1_partition_log_sample(const av1_partition_sample_t *sample);
void av1_partition_log_close(void);
#else
#define av1_partition_log_init(filename) ((void)0)
#define av1_partition_log_sample(sample) ((void)0)
#define av1_partition_log_close() ((void)0)
#endif

#endif
