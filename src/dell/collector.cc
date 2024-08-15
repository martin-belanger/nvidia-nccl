/**
 * Copyright (c) 2024, Dell Technologies Inc. or its subsidiaries.
 * All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Authors: Martin Belanger <Martin.Belanger@dell.com>
 */

#include <sys/shm.h>
#include <sys/mman.h>
#include <iostream>

#include "nccl.h"
#include "info.h"
#include "comm.h"
#include "debug.h"
#include "nccl_common.h"
#include "dell/collector.h"
#include "utils.h"

PRE_AND_POST_ENUM_INCREMENT(ncclFunc_t)
PRE_AND_POST_ENUM_INCREMENT(ncclRedOp_t)
PRE_AND_POST_ENUM_INCREMENT(ncclDataType_t)

static const char* func_name(ncclFunc_t func) {
	switch (func) {
	case ncclFuncBroadcast:     return "Broadcast";
	case ncclFuncReduce:        return "Reduce";
	case ncclFuncAllGather:     return "AllGather";
	case ncclFuncReduceScatter: return "ReduceScatter";
	case ncclFuncAllReduce:     return "AllReduce";
	case ncclFuncSendRecv:      return "SendRecv";
	case ncclFuncSend:          return "Send";
	case ncclFuncRecv:          return "Recv";
	default:;
	}
	return "unknown";
}

/**
 * Return the interval name corresponding to each counters of
 * pw2_counts[] (nccl_stats_c).
 *
 * The interval are specified in "Interval Notation" where:
 *
 * Parentheses, ( or ), are used to signify that an endpoint is
 * not included, called exclusive.
 *
 * Brackets, [ or ], are used to indicate that an endpoint is
 * included, called inclusive.
 *
 * @param pwr2_index: The index used to access pw2_counts[].
 *      	    This is the index previously calculated by
 *      	    pwr2_index().
 */
static const char* pwr2_interval(unsigned int pwr2_index)
{
	switch (pwr2_index) {
	case  0: return "count_0_1K"; /* pw2_counts[0] contains sizes from 0 to 1024 */
	case  1: return "count_1K_2K";
	case  2: return "count_2K_4K";
	case  3: return "count_4K_8K";
	case  4: return "count_8K_16K";
	case  5: return "count_16K_32K";
	case  6: return "count_32K_64K";
	case  7: return "count_64K_128K";
	case  8: return "count_128K_256K";
	case  9: return "count_256K_512K";
	case 10: return "count_512K_1M";
	case 11: return "count_1M_2M";
	case 12: return "count_2M_4M";
	case 13: return "count_4M_8M";
	case 14: return "count_8M_16M";
	case 15: return "count_16M_32M";
	case 16: return "count_32M_64M";
	case 17: return "count_64M_128M";
	case 18: return "count_128M_256M";
	case 19: return "count_256M_512M";
	case 20: return "count_512M_1G";
	case 21: return "count_1G_2G";
	case 22: return "count_2G_4G";
	case 23: return "count_4G_8G";
	case 24: return "count_8G_16G";
	case 25: return "count_16G_32G";
	case 26: return "count_32G_64G";
	case 27: return "count_64G_128G";
	case 28: return "count_128G_256G";
	case 29: return "count_256G_512G";
	case 30: return "count_512G_1T";
	/* pw2_counts[31] contains sizes 1T and above */
	default:;
	}

	return "count_1T_Inf";
}

/**
 * Determine in which bucket of pw2_counts[] (nccl_stats_c), in
 * other words which "index" of the pw2_counts[] array, the
 * value @count belongs to.
 *
 * We keep track of the different data sizes being processed by
 * NCCL. Each counter in pw2_counts[] correspond to a size
 * interval as a power of 2. For example:
 *
 * pw2_counts[0]  contains sizes from 0 to 1KB      (0 <= msb <= 9)
 * pw2_counts[1]  contains sizes from 1KB to 2KB    (msb == 10)
 * pw2_counts[2]  contains sizes from 2KB to 4KB    (msb == 11)
 * ...
 * pw2_counts[31] contains sizes from 1TB and above (40 <= msb <= 63)
 *
 * @param count: The data size being processed by NCCL.
 *
 * @return The index used to access pw2_counts[] (nccl_stats_c)
 */
static unsigned int pwr2_index(unsigned long long count)
{
	auto msb = log2i(count);
	return (msb <= 9) ? 0 : std::min((int)(msb - 9), (MAX_NUM_BINS - 1));
}

NCCL_PARAM(WorkloadId, "WORKLOAD_ID", 0); // This defines the function ncclParamWorkloadId()

/**
 * Build the string containing the name of the Shared Memory
 * device. This is the name that will appear under "/dev/shm/".
 * There is one shared memory per rank.
 *
 * @param rank: NCCL Rank
 *
 * @return The name of the shared memory.
 */
static std::string get_shmpath(int rank)
{
	unsigned int workloadId = ncclParamWorkloadId();

	return "/prometheus-nccl-" + std::to_string(rank) + "-" + std::to_string(workloadId);
}

std::ostream& operator<<(std::ostream  & stream_r, const nccl_data_v1_c  * data_p)
{
	stream_r << "  opers: \t" << data_p->opers  << '\n'
		 << "  count: \t" << data_p->count  << '\n'
		 << "  chunks:\t" << data_p->chunks << '\n';

	for (auto id = 0; id < MAX_NUM_BINS; id++) {
		if (data_p->pw2_counts[id])
			stream_r << "      bins[" << id << "]:\t" << data_p->pw2_counts[id] << '\t' << pwr2_interval(id) << '\n';
	}

	return stream_r;
}


/******************************************************************************/
/******************************************************************************/
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

dell_collector_c::dell_collector_c(int rank) : rank_m(rank)
{
	std::string shmpath = get_shmpath(rank_m);
	shm_unlink(shmpath.c_str()); // Ensure there is no leftover SHM from a previous run.

	fd_m = shm_open(shmpath.c_str(), O_CREAT | O_EXCL | O_RDWR, 0644);
	if (fd_m == -1) {
		WARN("Prometheus: Failed to create shared memory");
		return;
	}

	if (ftruncate(fd_m, shms_m) == -1) {
		WARN("Prometheus: Failed to resize shared memory");
		goto error;
	}

	/* Map the shm object into the caller's address space. */
	shmp_m = (nccl_stats_c *)mmap(NULL, shms_m, PROT_READ | PROT_WRITE, MAP_SHARED, fd_m, 0);
	if ((void *)shmp_m == MAP_FAILED) {
		WARN("Prometheus: Failed to mmap shared memory");
		goto error;
	}

	shmp_m->clear();

	return;

error:
	close(fd_m);
	shm_unlink(shmpath.c_str());

	fd_m = -1;
	shmp_m = (nccl_stats_c *)MAP_FAILED;
}

dell_collector_c::~dell_collector_c(void)
{
	if (shmp_m) {
		munmap(shmp_m, 0);
		shmp_m = nullptr;
	}

	if (fd_m != -1) {
		close(fd_m);
		fd_m = -1;
	}

	shm_unlink(get_shmpath(rank_m).c_str());
}

/**
 * Pretty all the counters collected to stdout.
 */
void dell_collector_c::print_counters()
{
	std::cout << "\n>>>>rank" << rank_m;

	for (auto ifunc = (ncclFunc_t)0; ifunc < ncclNumFuncs; ifunc++) {
		for (auto ioper = (ncclRedOp_t)0; ioper < ncclNumOps; ioper++) {
			for (auto itype = (ncclDataType_t)0; itype < ncclNumTypes; itype++) {
				for (auto ialgo = 0; ialgo < NCCL_NUM_ALGORITHMS; ialgo++) {
					for (auto iprot = 0; iprot < NCCL_NUM_PROTOCOLS; iprot++) {
						auto data_p = &shmp_m->data_m[ifunc][ioper][itype][ialgo][iprot];
						if (!data_p->empty()) {
							std::cout << "\nstats["
								  << func_name(ifunc)  << "]["
								  << ncclOpToString(ioper)    << "]["
								  << ncclDatatypeToString(itype)  << "]["
								  << ncclAlgoToString(ialgo)  << "]["
								  << ncclProtoToString(iprot) << "]:\n"
								  << data_p;
						}
					}
				}
			}
		}
	}

	std::cout << "<<<<rank" << rank_m << '\n';
}

void dell_collector_c::collect(const struct ncclInfo *info_p)
{
	if ((info_p->coll      < ncclNumFuncs) &&
	    (info_p->op        < ncclMaxRedOp) &&
	    (info_p->datatype  < ncclNumTypes) &&
	    (info_p->algorithm < NCCL_NUM_ALGORITHMS) &&
	    (info_p->protocol  < NCCL_NUM_PROTOCOLS)
	   ) {
		auto data_p = &shmp_m->data_m[info_p->coll][info_p->op][info_p->datatype][info_p->algorithm][info_p->protocol];

		data_p->opers++;
		data_p->count += info_p->count;
		data_p->chunks += info_p->chunkCount;
		data_p->pw2_counts[pwr2_index(info_p->count)]++;
	}
}

/******************************************************************************/
/******************************************************************************/
static struct ncclComm *ugly_kludge_comm = nullptr; // Ugly kludge (see destructor code below)

void dell_collector_start(struct ncclComm *comm) {
	if (!comm || comm->dell_collector)
		return;

	ugly_kludge_comm = comm;
	comm->dell_collector = new dell_collector_c(comm->rank);
}

void dell_collector_stop(struct ncclComm *comm) {
	if (comm && comm->dell_collector) {
		comm->dell_collector->print_counters();
		delete comm->dell_collector;
		comm->dell_collector = nullptr;
		ugly_kludge_comm = nullptr;
	}
}

void dell_collector_collect(struct ncclComm *comm, const struct ncclInfo *collInfo) {
	if (comm && comm->dell_collector && collInfo)
		comm->dell_collector->collect(collInfo);
}

// Unfortunately, many programmers simply exit applications without a care in
// the world for cleaning after themselves. One thing that we want to do after
// execution of this code is print all the stats. Since we cannot rely on
// users to call the cleanup code (i.e. commFree() which invokes
// dell_collector_stop() above), we have to resort to this ugly kludge.
void exit()__attribute__((destructor));
void exit() {
	if (ugly_kludge_comm) {
		dell_collector_stop(ugly_kludge_comm);
		ugly_kludge_comm = nullptr;
	}
}
