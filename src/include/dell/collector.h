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

class dell_collector_c {
public:
	dell_collector_c(int rank);
	~dell_collector_c(void);

	void collect(const struct ncclInfo *collInfo);
	void print_counters();

	const char *shm_data() { return (const char *)shmp_m; }
	size_t shm_size() { return shms_m; }

protected:
	int           rank_m  = -1;
	int           fd_m    = -1;       // Shared memory file descriptor
	size_t        shms_m  = SHM_SIZE; // Shared memory size
	nccl_stats_c *shmp_m  = (nccl_stats_c *)MAP_FAILED; // Pointer to shared memory (mmap)
};

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
