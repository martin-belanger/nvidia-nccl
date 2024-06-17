/**
 * Copyright (c) 2024, Dell Technologies Inc. or its subsidiaries.
 * All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Authors: Martin Belanger <Martin.Belanger@dell.com>
 *
 * Requires C++ headers for JSON support (<nlohmann/json.hpp>):
 *   sudo apt-get install -y nlohmann-json3-dev
 */

#include <sys/shm.h>
#include <sys/mman.h>
#include <iostream>

#include "nccl.h"
#include "info.h"
#include "comm.h"
#include "debug.h"
#include "nccl_common.h"
#include "dell/prometheus.h"

#define CALCULATE_BYTES 0

/**
 * Calculates the smallest integral power of two that is not
 * smaller than @value.
 */
static unsigned long long ceil2(size_t value) {
	return 1ULL << (log2i(value) + 1);
}

#if (CALCULATE_BYTES)
static size_t data_size(ncclDataType_t data_type) {
	switch (data_type) {
	case ncclInt8:     return 1;
	case ncclUint8:    return 1;
	case ncclInt32:    return 4;
	case ncclUint32:   return 4;
	case ncclInt64:    return 8;
	case ncclUint64:   return 8;
	case ncclFloat16:  return 2;
	case ncclFloat32:  return 4;
	case ncclFloat64:  return 8;
	case ncclBfloat16: return 2;
	default:;
	}
	return 0;
}
#endif // CALCULATE_BYTES

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
 * Return the interval in which a value falls using Interval
 * Notation where:
 *
 * Parentheses, ( or ), are used to signify that an endpoint is
 * not included, called exclusive.
 *
 * Brackets, [ or ], are used to indicate that an endpoint is
 * included, called inclusive.
 */
static const char* pwr2_interval(unsigned long long value)
{
	switch (log2i(value)) {
	case 0 ... 9: return "[0,1K)";
	case 10: return "[1K,2K)";
	case 11: return "[2K,4K)";
	case 12: return "[4K,8K)";
	case 13: return "[8K,16K)";
	case 14: return "[16K,32K)";
	case 15: return "[32K,64K)";
	case 16: return "[64K,128K)";
	case 17: return "[128K,256K)";
	case 18: return "[256K,512K)";
	case 19: return "[512K,1M)";
	case 20: return "[1M,2M)";
	case 21: return "[2M,4M)";
	case 22: return "[4M,8M)";
	case 23: return "[8M,16M)";
	case 24: return "[16M,32M)";
	case 25: return "[32M,64M)";
	case 26: return "[64M,128M)";
	case 27: return "[128M,256M)";
	case 28: return "[256M,512M)";
	case 29: return "[512M,1G)";
	case 30: return "[1G,2G)";
	case 31: return "[2G,4G)";
	default:;
	}

	return "[4G,Inf)";
}

/******************************************************************************/
/******************************************************************************/

static std::string get_shmpath(int rank)
{
	return "/prometheus-nccl-" + std::to_string(rank);
}

void prometheus_shm_open(struct ncclComm *comm) {
	if (comm->prometheus) {
		delete comm->prometheus;
		comm->prometheus = NULL;
	}

	std::string shmpath = get_shmpath(comm->rank);
	shm_unlink(shmpath.c_str());

	prometheus_t *prometheus_p = new prometheus_t;
	prometheus_p->shms = SMH_INITIAL_SIZE;

	prometheus_p->fd = shm_open(shmpath.c_str(), O_CREAT | O_EXCL | O_RDWR, 0644);
	if (prometheus_p->fd == -1) {
		WARN("Prometheus: Failed to create shared memory");
		goto error1;
	}

	if (-1 == ftruncate(prometheus_p->fd, prometheus_p->shms)) {
		WARN("Prometheus: failed to resize shared memory");
		return;
	}

	/* Map the shm object into the caller's address space. */
	prometheus_p->shmp = mmap(NULL, prometheus_p->shms, PROT_READ | PROT_WRITE, MAP_SHARED, prometheus_p->fd, 0);
	if (prometheus_p->shmp == MAP_FAILED) {
		WARN("Prometheus: Failed to mmap shared memory");
		goto error2;
	}

	comm->prometheus = prometheus_p;
	return;

error2:
	close(prometheus_p->fd);
	shm_unlink(shmpath.c_str());

error1:
	delete prometheus_p;
	comm->prometheus = NULL;
}

void prometheus_shm_close(struct ncclComm *comm) {
	struct prometheus *prometheus_p = comm->prometheus;

	comm->prometheus = NULL;
	if (prometheus_p) {
		// Dump collected data to terminal
		std::cout << "Stats for Rank " << comm->rank << '\n';
		std::cout << std::setw(3) << prometheus_p->json << '\n';

		if (prometheus_p->shmp) {
			munmap(prometheus_p->shmp, 0);
			prometheus_p->shmp = NULL;
		}

		if (prometheus_p->fd != -1) {
			close(prometheus_p->fd);
			prometheus_p->fd = -1;
		}

		delete prometheus_p;
	}

	std::string shmpath = get_shmpath(comm->rank);
	shm_unlink(shmpath.c_str());
}

void prometheus_collect(struct ncclComm *comm, const struct ncclInfo *collInfo) {
	prometheus_t *prometheus_p = comm->prometheus;

	if (prometheus_p && prometheus_p->shmp) {
		auto func  = func_name(collInfo->coll);
		auto op    = op_name(collInfo->op);
		auto type  = type_name(collInfo->datatype);
		auto algo  = algo_name(collInfo->algorithm);
		auto proto = proto_name(collInfo->protocol);

		// Retrieve the stats object, or create a new object if it doesn't exist.
		auto stats = prometheus_p->json.value(func, nlohmann::json({})).value(op, nlohmann::json({})).value(type, nlohmann::json({})).value(algo, nlohmann::json({})).value(proto, nlohmann::json({}));

		// Increment the total count
		auto count = stats.value("count", nlohmann::json(0ULL)).get<unsigned long long>() + collInfo->count;
		stats["count"] = count;

		// Calculate and increment byte count
#if (CALCULATE_BYTES)
		stats["bytes"] = stats.value("bytes", nlohmann::json(0ULL)).get<unsigned long long>() + (count * collInfo->chunkCount * data_size(collInfo->datatype));
#endif // CALCULATE_BYTES
		stats["opers"] = stats.value("opers", nlohmann::json(0ULL)).get<unsigned long long>() + 1ULL;
		stats["chunks"] = stats.value("chunks", nlohmann::json(0ULL)).get<unsigned long long>() + collInfo->chunkCount;

		// Increment "power of 2" counters
		auto pwr2id = std::string("count-") + pwr2_interval(collInfo->count);
		auto pwr2val = stats.value(pwr2id, nlohmann::json(0ULL)).get<unsigned long long>() + 1ULL;
		stats[pwr2id] = pwr2val;

		prometheus_p->json[func][op][type][algo][proto] = stats;

		// Write JSON to shared memory
		auto text = prometheus_p->json.dump();

		// Adjust shared memory size to fit JSON text (if needed)
		if (text.length() > prometheus_p->shms) {
			// New size is rounded up to next power of 2.
			auto new_size = ceil2(text.length());

			if (new_size > SMH_MAX_SIZE) {
				WARN("Prometheus: Shared memory required is too big sz=%ld (max: %dM)", text.length(), SMH_MAX_SIZE/(1024*1024));
				return;
			}

			if (-1 == ftruncate(prometheus_p->fd, new_size)) {
				WARN("Prometheus: failed to resize shared memory sz=%lld", new_size);
				return;
			}

			prometheus_p->shmp = mremap(prometheus_p->shmp, prometheus_p->shms,
						    new_size, MREMAP_MAYMOVE, NULL);
			if (prometheus_p->shmp == MAP_FAILED) {
				WARN("Prometheus: Failed to mremap shared memory sz=%lld", new_size);
				return;
			}

			prometheus_p->shms = new_size;
		}

		memcpy(prometheus_p->shmp, text.c_str(), text.length() + 1);
	}
}
