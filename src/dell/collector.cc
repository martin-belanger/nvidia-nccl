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

static const char* op_name(ncclRedOp_t op) {
	switch (op) {
	case ncclSum:  return "Sum";
	case ncclProd: return "Prod";
	case ncclMax:  return "Max";
	case ncclMin:  return "Min";
	case ncclAvg:  return "Avg";
	default:;
	}
	return "unknown";
}

static const char* type_name(ncclDataType_t type) {
	switch (type) {
	case ncclInt8:     return "int8";
	case ncclUint8:    return "uint8";
	case ncclInt32:    return "int32";
	case ncclUint32:   return "uint32";
	case ncclInt64:    return "int64";
	case ncclUint64:   return "uint64";
	case ncclFloat16:  return "float16";
	case ncclFloat32:  return "float32";
	case ncclFloat64:  return "float64";
	case ncclBfloat16: return "bfloat16";
	default:;
	}
	return "unknown";
}

static const char* proto_name(int proto) {
	switch (proto) {
	case NCCL_PROTO_LL:     return "LL";
	case NCCL_PROTO_LL128:  return "LL128";
	case NCCL_PROTO_SIMPLE: return "Simple";
	default:;
	}
	return "unknown";
}

static const char* algo_name(int algo) {
	switch (algo) {
	case NCCL_ALGO_TREE:           return "Tree";
	case NCCL_ALGO_RING:           return "Ring";
	case NCCL_ALGO_COLLNET_DIRECT: return "CollNetDirect";
	case NCCL_ALGO_COLLNET_CHAIN:  return "CollNetChain";
	case NCCL_ALGO_NVLS:           return "Nvls";
	case NCCL_ALGO_NVLS_TREE:      return "NvlsTree";
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
	case  0: return "[0,1K)"; /* pw2_counts[0] contains sizes from 0 to 1024 */
	case  1: return "[1K,2K)";
	case  2: return "[2K,4K)";
	case  3: return "[4K,8K)";
	case  4: return "[8K,16K)";
	case  5: return "[16K,32K)";
	case  6: return "[32K,64K)";
	case  7: return "[64K,128K)";
	case  8: return "[128K,256K)";
	case  9: return "[256K,512K)";
	case 10: return "[512K,1M)";
	case 11: return "[1M,2M)";
	case 12: return "[2M,4M)";
	case 13: return "[4M,8M)";
	case 14: return "[8M,16M)";
	case 15: return "[16M,32M)";
	case 16: return "[32M,64M)";
	case 17: return "[64M,128M)";
	case 18: return "[128M,256M)";
	case 19: return "[256M,512M)";
	case 20: return "[512M,1G)";
	case 21: return "[1G,2G)";
	case 22: return "[2G,4G)";
	case 23: return "[4G,8G)";
	case 24: return "[8G,16G)";
	case 25: return "[16G,32G)";
	case 26: return "[32G,64G)";
	case 27: return "[64G,128G)";
	case 28: return "[128G,256G)";
	case 29: return "[256G,512G)";
	case 30: return "[512G,1T)";
	/* pw2_counts[31] contains sizes 1T and above */
	default:;
	}

	return "[1T,Inf)";
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
	return "/prometheus-nccl-" + std::to_string(rank);
}

std::ostream& operator<<(std::ostream  & stream_r, const nccl_data_v1_c  * data_p)
{
	stream_r << "  opers: \t" << data_p->opers  << '\n'
		 << "  count: \t" << data_p->count  << '\n'
		 << "  chunks:\t" << data_p->chunks << '\n';

	for (auto id = 0; id < MAX_NUM_BINS; id++) {
		if (data_p->pw2_counts[id])
			stream_r << "  bins[" << id << "]:\t" << data_p->pw2_counts[id] << '\t' << pwr2_interval(id) << '\n';
	}

	return stream_r;
}

/**
 * Pretty all the counters collected to stdout.
 */
void dell_collector_c::print_counters()
{
	std::cout << "=====================================================\n"
		  << "NCCL stats for rank " << rank_m << '\n'
		  << "=====================================================\n";

	for (auto ifunc = (ncclFunc_t)0; ifunc < ncclNumFuncs; ifunc++) {
		for (auto ioper = (ncclRedOp_t)0; ioper < ncclNumOps; ioper++) {
			for (auto itype = (ncclDataType_t)0; itype < ncclNumTypes; itype++) {
				for (auto ialgo = 0; ialgo < NCCL_NUM_ALGORITHMS; ialgo++) {
					for (auto iprot = 0; iprot < NCCL_NUM_PROTOCOLS; iprot++) {
						auto data_p = &shmp_m->data_m[ifunc][ioper][itype][ialgo][iprot];
						if (!data_p->empty()) {
							std::cout << "stats["
								  << func_name(ifunc)  << "]["
								  << op_name(ioper)    << "]["
								  << type_name(itype)  << "]["
								  << algo_name(ialgo)  << "]["
								  << proto_name(iprot) << "]:\n"
								  << data_p
								  << '\n';
						}
					}
				}
			}
		}
	}
	std::cout << '\n';
}

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
void dell_collector_start(struct ncclComm *comm) {
	if (!comm || comm->dell_collector)
		return;

	comm->dell_collector = new dell_collector_c(comm->rank);
}

void dell_collector_stop(struct ncclComm *comm) {
	if (comm && comm->dell_collector) {
		comm->dell_collector->print_counters();
		delete comm->dell_collector;
		comm->dell_collector = nullptr;
	}
}

void dell_collector_collect(struct ncclComm *comm, const struct ncclInfo *collInfo) {
	if (comm && comm->dell_collector && collInfo)
		comm->dell_collector->collect(collInfo);
}
