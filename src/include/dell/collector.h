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

#include <string.h>  // memset()

#define SHM_INITIAL_SIZE (64 * 1024)
#define SHM_MAX_SIZE     (4 * 1024 * 1024)

#define WITH_BYTES       0
#define MAX_NUM_BINS     32

class nccl_stats_c {
public:
	bool empty() const { return opers == 0; }
	std::string to_string() const;

	unsigned long long opers;
	unsigned long long count;
	unsigned long long chunks;
	unsigned long long pw2_counts[MAX_NUM_BINS];

#if (WITH_BYTES)
	unsigned long long bytes;
#endif // WITH_BYTES

};

std::ostream& operator<<(std::ostream  & stream_r, const nccl_stats_c  * stats_p);

class kv_obj_c
{
public:
	void clear() {
		val_m.clear();
		sep_m.clear();
	}
	bool empty() const { return val_m.empty(); }
	void append(const char * key, const std::string & val_r);
	void append(const char * key, const kv_obj_c & val_r) {append(key, val_r.val_m);}

	const std::string & str() const { return val_m; }

protected:
	std::string   val_m;
	std::string   sep_m;
};

class dell_collector_c {
public:
	dell_collector_c(int rank);
	~dell_collector_c(void);

	void collect(const struct ncclInfo *collInfo);
	std::string stringify();
	void print_counters();

	void shm_save();
	void shm_unlink();
	const char *shm_data() { return (const char *)shmp_m; }
	size_t shm_size() { return shms_m; }
	bool shm_resize(size_t length);

protected:
	int           rank_m  = -1;
	int           fd_m    = -1;               // Shared memory file descriptor
	size_t        shms_m  = SHM_INITIAL_SIZE; // Shared memory size
	void         *shmp_m  = MAP_FAILED;       // Pointer to shared memory (mmap)
	nccl_stats_c  stats_m[ncclNumFuncs][ncclNumOps][ncclNumTypes][NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS] = {0};
	bool          shm_max_reached_m = false;
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
