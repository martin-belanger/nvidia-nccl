#ifndef _DELL_COLLECTOR_H_
#define _DELL_COLLECTOR_H_
/**
 * Copyright (c) 2024, Dell Technologies Inc. or its
 * subsidiaries. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Authors: Martin Belanger <Martin.Belanger@dell.com>
 */

#ifdef DELL_COLLECTOR

#define MAX_NUM_BINS  32
#define SHM_VERSION   1

struct nccl_data_v1_c {
	bool empty() const { return opers == 0; }

	unsigned long long opers;
	unsigned long long count;
	unsigned long long chunks;
	unsigned long long pw2_counts[MAX_NUM_BINS];
};

struct nccl_stats_c {
	void clear() {
		memset((void *)this, 0, sizeof(*this));
		version_m = SHM_VERSION;
	}

	unsigned long long  version_m;
	nccl_data_v1_c      data_m[ncclNumFuncs][ncclNumOps][ncclNumTypes][NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
};

#define SHM_SIZE  sizeof(nccl_stats_c)

std::ostream& operator<<(std::ostream  & stream_r, const nccl_data_v1_c  * stat_p);

/**
 * Allocate and initialize Shared Memory for Prometheus metrics.
 * The Shared Memory is used to exchange metrics with a
 * Prometheus agent.
 */
void dell_collector_start(struct ncclComm *comm);

/**
 * Free Prometheus Shared Memory.
 */
void dell_collector_stop(struct ncclComm *comm);

/**
 * Collect Prometheus metrics and update Shared Memory.
 */
void dell_collector_collect(struct ncclComm *comm, const struct ncclInfo *collInfo);

#else  // DELL_COLLECTOR

#define dell_collector_start(...)   do {} while(0)
#define dell_collector_stop(...)    do {} while(0)
#define dell_collector_collect(...) do {} while(0)

#endif // DELL_COLLECTOR


// Helper macro to define enum pre/post increment functions (++enum / enum++).
// This keeps C++ happy and avoids the trouble of casting enums back-and-forth
// to/from int.
#define PRE_AND_POST_ENUM_INCREMENT(ENUM) \
inline ENUM& operator++(ENUM& enumVal) { \
   return (ENUM&)(++(int&)enumVal); \
} \
\
inline ENUM operator++(ENUM& enumVal, int postIncrement) { \
   ENUM oldValue = enumVal; \
   ++enumVal; \
   return oldValue; \
}

#endif // _DELL_COLLECTOR_H_
