#ifndef _DELL_PROMETHEUS_H_
#define _DELL_PROMETHEUS_H_
/**
 * Copyright (c) 2024, Dell Technologies Inc. or its
 * subsidiaries. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Authors: Martin Belanger <Martin.Belanger@dell.com>
 */

#ifdef DELL_PROMETHEUS

#include <nlohmann/json.hpp>

#define SMH_INITIAL_SIZE 4096
#define SMH_MAX_SIZE     (2 * 1024 * 1024)

struct prometheus {
	prometheus() : fd(-1), shms(SMH_INITIAL_SIZE), shmp(nullptr), json({}) {}

	int              fd;   // Shared memory file descriptor
	size_t           shms; // Shared memory size
	void            *shmp; // Pointer to shared memory (mmap)
	nlohmann::json   json;
};
typedef struct prometheus prometheus_t;


/**
 * Allocate and initialize Shared Memory for Prometheus metrics.
 * The Shared Memory is used to exchange metrics with a
 * Prometheus agent.
 */
void prometheus_shm_open(struct ncclComm *comm);

/**
 * Free Prometheus Shared Memory.
 */
void prometheus_shm_close(struct ncclComm *comm);

/**
 * Collect Prometheus metrics and update Shared Memory.
 */
void prometheus_collect(struct ncclComm *comm, const struct ncclInfo *collInfo);

#else  // DELL_PROMETHEUS

#define prometheus_shm_open(...)  do {} while(0)
#define prometheus_shm_close(...) do {} while(0)
#define prometheus_collect(...)   do {} while(0)

#endif // DELL_PROMETHEUS

#endif // _DELL_PROMETHEUS_H_
