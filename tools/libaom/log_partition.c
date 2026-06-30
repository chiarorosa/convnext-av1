#ifdef LOG_PARTITION_DATA

#include "log_partition.h"

#include <stdlib.h>

FILE *av1_partition_log_file = NULL;

void av1_partition_log_init(const char *filename) {
  av1_partition_log_file = fopen(filename, "wb");
  if (!av1_partition_log_file) {
    perror("failed to open AV1 partition log");
    exit(EXIT_FAILURE);
  }
}

void av1_partition_log_sample(const av1_partition_sample_t *sample) {
  if (av1_partition_log_file && sample) {
    fwrite(sample, sizeof(av1_partition_sample_t), 1, av1_partition_log_file);
  }
}

void av1_partition_log_close(void) {
  if (av1_partition_log_file) {
    fclose(av1_partition_log_file);
    av1_partition_log_file = NULL;
  }
}

#endif
