//
// Created by MiseTy on 2023/1/31.
//
#include "ExtractPointCloud.h"
#include "TsdfFusion.h"
#include "cx.h"

enum {
    CTA_SIZE_X = 32,
    CTA_SIZE_Y = 6,
    CTA_SIZE = CTA_SIZE_X * CTA_SIZE_Y,

    MAX_LOCAL_POINTS = 3
};

__device__ int global_count = 0;
__device__ int output_count;
__device__ unsigned int blocks_done = 0;

__shared__ float storage_X[CTA_SIZE * MAX_LOCAL_POINTS];
__shared__ float storage_Y[CTA_SIZE * MAX_LOCAL_POINTS];
__shared__ float storage_Z[CTA_SIZE * MAX_LOCAL_POINTS];


struct Scanner {
    int3 volume_resolution;//	resolution of the volume
    float voxel_size;
    mutable PtrSz<float3> output;
    mutable PtrSz<float3> grad;

    PtrStep<float> value_volume;
    PtrStep<int> weight_volume;
    PtrStep<float> grad_volume;

    __device__ __forceinline__ float fetch(int x, int y, int z, int &w) const {
        //	calculate the storage cell
        x = x % volume_resolution.x;
        y = y % volume_resolution.y;
        z = z % volume_resolution.z;
        devComplex res;
        unpack_tsdf(value_volume.ptr(volume_resolution.y * z + y)[x],
                    weight_volume.ptr(volume_resolution.y * z + y)[x],
                    grad_volume.ptr(volume_resolution.y * z + y)[x], res, w);
        // res += 1e-5f;
        return res.real();
    }
    __device__ __forceinline__ void
    operator()() const {
        int x = threadIdx.x + blockIdx.x * CTA_SIZE_X;
        int y = threadIdx.y + blockIdx.y * CTA_SIZE_Y;

        if (__all_sync(__activemask(), x >= volume_resolution.x) || __all_sync(__activemask(), y >= volume_resolution.y))
            return;

        float3 V;
        V.x = (x + 0.5f) * voxel_size;
        V.y = (y + 0.5f) * voxel_size;

        int ftid = cx::Block::flattenedThreadId();

        for (int z = 0; z < volume_resolution.z - 1; ++z) {
            float3 points[MAX_LOCAL_POINTS];

            int local_count = 0;

            if (x < volume_resolution.x - 1 && y < volume_resolution.y - 1) {
                int W;
                float F = fetch(x, y, z, W);

                if (F < 0.99f) {
                    V.z = (z + 0.5f) * voxel_size;

                    //process dx
                    if (x + 1 < volume_resolution.x) {
                        int Wn;
                        float Fn = fetch(x + 1, y, z, Wn);
                        if (Fn < 0.99f)
                            if ((F > 0 && Fn < 0) || (F < 0 && Fn > 0)) {
                                float3 p;
                                p.y = V.y;
                                p.z = V.z;
                                p.x = V.x - (F / (Fn - F)) * voxel_size;
                                points[local_count++] = p;
                            }
                    }

                    //process dy
                    if (y + 1 < volume_resolution.y) {
                        int Wn;
                        float Fn = fetch(x, y + 1, z, Wn);
                        if (Fn < 0.99f)
                            if ((F > 0 && Fn < 0) || (F < 0 && Fn > 0)) {
                                float3 p;
                                p.x = V.x;
                                p.z = V.z;

                                p.y = V.y - (F / (Fn - F)) * voxel_size;
                                points[local_count++] = p;
                            }
                    } /*  if (y + 1 < VOLUME_Y) */

                    //process dz
                    //if (z + 1 < VOLUME_Z) // guaranteed by loop
                    {
                        int Wn;
                        float Fn = fetch(x, y, z + 1, Wn);
                        if (Fn < 0.99f)
                            if ((F > 0 && Fn < 0) || (F < 0 && Fn > 0)) {
                                float3 p;
                                p.x = V.x;
                                p.y = V.y;
                                p.z = V.z - (F / (Fn - F)) * voxel_size;
                                points[local_count++] = p;
                            }
                    } /* if (z + 1 < VOLUME_Z) */
                }     /* if (W != 0 && F != 1.f1) */
            }         /* if (x < VOLUME_X && y < VOLUME_Y) */

            int total_warp = __popc(__ballot_sync(__activemask(), local_count > 0)) +
                             __popc(__ballot_sync(__activemask(), local_count > 1)) +
                             __popc(__ballot_sync(__activemask(), local_count > 2));


            if (total_warp > 0) {
                int lane = cx::Warp::laneId();
                int storage_index = (ftid >> cx::Warp::LOG_WARP_SIZE) * cx::Warp::WARP_SIZE * MAX_LOCAL_POINTS;

                volatile int *cta_buffer = (int *) (storage_X + storage_index);

                cta_buffer[lane] = local_count;
                int offset = scan_warp<exclusive>(cta_buffer, lane);

                if (lane == 0) {
                    int old_global_count = atomicAdd(&global_count, total_warp);
                    cta_buffer[0] = old_global_count;
                }
                int old_global_count = cta_buffer[0];

                for (int l = 0; l < local_count; ++l) {
                    storage_X[storage_index + offset + l] = points[l].x;
                    storage_Y[storage_index + offset + l] = points[l].y;
                    storage_Z[storage_index + offset + l] = points[l].z;
                }

                float3 *pos = output.data + old_global_count + lane;
                //                printf("%d, %d\n", old_global_count, lane);
                for (int idx = lane; idx < total_warp; idx += cx::Warp::STRIDE, pos += cx::Warp::STRIDE) {
                    float x = storage_X[storage_index + idx];
                    float y = storage_Y[storage_index + idx];
                    float z = storage_Z[storage_index + idx];
                    store_point_type(x, y, z, pos);
                }

                bool full = (old_global_count + total_warp) >= output.size;

                if (full)
                    break;
            }
        }


        ///////////////////////////
        // prepare for future scans
        if (ftid == 0) {
            unsigned int total_blocks = gridDim.x * gridDim.y * gridDim.z;
            unsigned int value = atomicInc(&blocks_done, total_blocks);

            //last block
            if (value == total_blocks - 1) {
                output_count = min((int) output.size, global_count);
                blocks_done = 0;
                global_count = 0;
            }
        }
    } /* operator() */

    __device__ __forceinline__ void
    store_point_type(float x, float y, float z, float3 *ptr) const {
        *ptr = make_float3(x, y, z);
    }
};
__global__ void
extractKernel(const Scanner sn) {
    sn();
}


size_t extractPoints(const PtrStep<float> &value_volume, const PtrStep<int> &weight_volume,
                     const PtrStep<float> &grad_volume, const int3 &volume_resolution,
                     float voxel_size, PtrSz<float3> output) {
    Scanner sn;
    sn.value_volume = value_volume;
    sn.weight_volume = weight_volume;

    sn.grad_volume = grad_volume;
    sn.volume_resolution = volume_resolution;
    sn.voxel_size = voxel_size;
    sn.output = output;


    dim3 block(CTA_SIZE_X, CTA_SIZE_Y);
    dim3 grid(divUp(volume_resolution.x, block.x), divUp(volume_resolution.y, block.y));
    extractKernel<<<grid, block>>>(sn);
    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaDeviceSynchronize());

    int size;
    cudaSafeCall(cudaMemcpyFromSymbol(&size, output_count, sizeof(size)));
    return (std::size_t) size;
}


struct ExtractNormals {
    int3 volume_resolution;//	resolution of the volume
    float voxel_size;

    PtrStep<float> value_volume;
    PtrStep<int> weight_volume;
    PtrStep<float> grad_volume;

    PtrSz<float3> points;

    mutable float3 *output;

    __device__ __forceinline__ float readTsdf(int x, int y, int z) const {
        //	calculate the storage cell
        x = x % volume_resolution.x;
        y = y % volume_resolution.y;
        z = z % volume_resolution.z;
        devComplex res;
        int w;
        unpack_tsdf(value_volume.ptr(volume_resolution.y * z + y)[x],
                    weight_volume.ptr(volume_resolution.y * z + y)[x],
                    grad_volume.ptr(volume_resolution.y * z + y)[x], res, w);
        // res += 1e-5f;
        return res.real();
    }

    __device__ __forceinline__ float3
    fetchPoint(int idx) const {
        float3 p = points.data[idx];
        return make_float3(p.x, p.y, p.z);
    }
    __device__ __forceinline__ void
    storeNormal(int idx, float3 normal) const {
        float3 n;
        n.x = normal.x;
        n.y = normal.y;
        n.z = normal.z;
        output[idx] = n;
    }

    __device__ __forceinline__ int3
    getVoxel(const float3 &point) const {
        int vx = __float2int_rd(point.x / voxel_size);// round to negative infinity
        int vy = __float2int_rd(point.y / voxel_size);
        int vz = __float2int_rd(point.z / voxel_size);

        return make_int3(vx, vy, vz);
    }

    __device__ __forceinline__ void
    operator()() const {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;

        if (idx >= points.size)
            return;
        float3 n = make_float3(0.0f, 0.0f, 0.0f);

        float3 point = fetchPoint(idx);
        int3 g = getVoxel(point);

        if (g.x > 1 && g.y > 1 && g.z > 1 && g.x < volume_resolution.x - 2 && g.y < volume_resolution.y - 2 && g.z < volume_resolution.z - 2) {
            float3 t;

            t = point;
            t.x += voxel_size;
            float Fx1 = interpolateTrilineary(t);

            t = point;
            t.x -= voxel_size;
            float Fx2 = interpolateTrilineary(t);

            n.x = (Fx1 - Fx2);

            t = point;
            t.y += voxel_size;
            float Fy1 = interpolateTrilineary(t);

            t = point;
            t.y -= voxel_size;
            float Fy2 = interpolateTrilineary(t);

            n.y = (Fy1 - Fy2);

            t = point;
            t.z += voxel_size;
            float Fz1 = interpolateTrilineary(t);

            t = point;
            t.z -= voxel_size;
            float Fz2 = interpolateTrilineary(t);

            n.z = (Fz1 - Fz2);
            float norm = n.x * n.x + n.y * n.y + n.z * n.z;
            n = make_float3(n.x / norm, n.y / norm, n.z / norm);
        }
        storeNormal(idx, n);
    }

    __device__ __forceinline__ float
    interpolateTrilineary(const float3 &point) const {
        int3 g = getVoxel(point);

        float vx = (g.x + 0.5f) * voxel_size;
        float vy = (g.y + 0.5f) * voxel_size;
        float vz = (g.z + 0.5f) * voxel_size;

        g.x = (point.x < vx) ? (g.x - 1) : g.x;
        g.y = (point.y < vy) ? (g.y - 1) : g.y;
        g.z = (point.z < vz) ? (g.z - 1) : g.z;

        float a = (point.x - (g.x + 0.5f) * voxel_size) / voxel_size;
        float b = (point.y - (g.y + 0.5f) * voxel_size) / voxel_size;
        float c = (point.z - (g.z + 0.5f) * voxel_size) / voxel_size;

        float res = readTsdf(g.x + 0, g.y + 0, g.z + 0) * (1 - a) * (1 - b) * (1 - c) +
                    readTsdf(g.x + 0, g.y + 0, g.z + 1) * (1 - a) * (1 - b) * c +
                    readTsdf(g.x + 0, g.y + 1, g.z + 0) * (1 - a) * b * (1 - c) +
                    readTsdf(g.x + 0, g.y + 1, g.z + 1) * (1 - a) * b * c +
                    readTsdf(g.x + 1, g.y + 0, g.z + 0) * a * (1 - b) * (1 - c) +
                    readTsdf(g.x + 1, g.y + 0, g.z + 1) * a * (1 - b) * c +
                    readTsdf(g.x + 1, g.y + 1, g.z + 0) * a * b * (1 - c) +
                    readTsdf(g.x + 1, g.y + 1, g.z + 1) * a * b * c;
        return res;
    }
};

__global__ void
extractNormalsKernel(const ExtractNormals en) {
    en();
}


void extractNormals(const PtrStep<float> &value_volume, const PtrStep<int> &weight_volume, const PtrStep<float> &grad_volume, const int3 &volume_resolution,
                    float voxel_size, PtrSz<float3> points, PtrSz<float3> normal) {
    ExtractNormals en;
    en.value_volume = value_volume;
    en.weight_volume = weight_volume;
    en.grad_volume = grad_volume;
    en.volume_resolution = volume_resolution;
    en.voxel_size = voxel_size;
    en.points = points;
    en.output = normal.data;

    dim3 block(256);
    dim3 grid(divUp(points.size, block.x));

    extractNormalsKernel<<<grid, block>>>(en);
    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaDeviceSynchronize());
}

__global__ void extractMeshKernal(PtrStep<float> value_volume, int3 volume_resolution, float voxel_size, PtrSz<float3> points, int *indices) {
    const unsigned long long triangles[256] = {
            0ULL,
            33793ULL,
            36945ULL,
            159668546ULL,
            18961ULL,
            144771090ULL,
            5851666ULL,
            595283255635ULL,
            20913ULL,
            67640146ULL,
            193993474ULL,
            655980856339ULL,
            88782242ULL,
            736732689667ULL,
            797430812739ULL,
            194554754ULL,
            26657ULL,
            104867330ULL,
            136709522ULL,
            298069416227ULL,
            109224258ULL,
            8877909667ULL,
            318136408323ULL,
            1567994331701604ULL,
            189884450ULL,
            350847647843ULL,
            559958167731ULL,
            3256298596865604ULL,
            447393122899ULL,
            651646838401572ULL,
            2538311371089956ULL,
            737032694307ULL,
            29329ULL,
            43484162ULL,
            91358498ULL,
            374810899075ULL,
            158485010ULL,
            178117478419ULL,
            88675058979ULL,
            433581536604804ULL,
            158486962ULL,
            649105605635ULL,
            4866906995ULL,
            3220959471609924ULL,
            649165714851ULL,
            3184943915608436ULL,
            570691368417972ULL,
            595804498035ULL,
            124295042ULL,
            431498018963ULL,
            508238522371ULL,
            91518530ULL,
            318240155763ULL,
            291789778348404ULL,
            1830001131721892ULL,
            375363605923ULL,
            777781811075ULL,
            1136111028516116ULL,
            3097834205243396ULL,
            508001629971ULL,
            2663607373704004ULL,
            680242583802939237ULL,
            333380770766129845ULL,
            179746658ULL,
            42545ULL,
            138437538ULL,
            93365810ULL,
            713842853011ULL,
            73602098ULL,
            69575510115ULL,
            23964357683ULL,
            868078761575828ULL,
            28681778ULL,
            713778574611ULL,
            250912709379ULL,
            2323825233181284ULL,
            302080811955ULL,
            3184439127991172ULL,
            1694042660682596ULL,
            796909779811ULL,
            176306722ULL,
            150327278147ULL,
            619854856867ULL,
            1005252473234484ULL,
            211025400963ULL,
            36712706ULL,
            360743481544788ULL,
            150627258963ULL,
            117482600995ULL,
            1024968212107700ULL,
            2535169275963444ULL,
            4734473194086550421ULL,
            628107696687956ULL,
            9399128243ULL,
            5198438490361643573ULL,
            194220594ULL,
            104474994ULL,
            566996932387ULL,
            427920028243ULL,
            2014821863433780ULL,
            492093858627ULL,
            147361150235284ULL,
            2005882975110676ULL,
            9671606099636618005ULL,
            777701008947ULL,
            3185463219618820ULL,
            482784926917540ULL,
            2900953068249785909ULL,
            1754182023747364ULL,
            4274848857537943333ULL,
            13198752741767688709ULL,
            2015093490989156ULL,
            591272318771ULL,
            2659758091419812ULL,
            1531044293118596ULL,
            298306479155ULL,
            408509245114388ULL,
            210504348563ULL,
            9248164405801223541ULL,
            91321106ULL,
            2660352816454484ULL,
            680170263324308757ULL,
            8333659837799955077ULL,
            482966828984116ULL,
            4274926723105633605ULL,
            3184439197724820ULL,
            192104450ULL,
            15217ULL,
            45937ULL,
            129205250ULL,
            129208402ULL,
            529245952323ULL,
            169097138ULL,
            770695537027ULL,
            382310500883ULL,
            2838550742137652ULL,
            122763026ULL,
            277045793139ULL,
            81608128403ULL,
            1991870397907988ULL,
            362778151475ULL,
            2059003085103236ULL,
            2132572377842852ULL,
            655681091891ULL,
            58419234ULL,
            239280858627ULL,
            529092143139ULL,
            1568257451898804ULL,
            447235128115ULL,
            679678845236084ULL,
            2167161349491220ULL,
            1554184567314086709ULL,
            165479003923ULL,
            1428768988226596ULL,
            977710670185060ULL,
            10550024711307499077ULL,
            1305410032576132ULL,
            11779770265620358997ULL,
            333446212255967269ULL,
            978168444447012ULL,
            162736434ULL,
            35596216627ULL,
            138295313843ULL,
            891861543990356ULL,
            692616541075ULL,
            3151866750863876ULL,
            100103641866564ULL,
            6572336607016932133ULL,
            215036012883ULL,
            726936420696196ULL,
            52433666ULL,
            82160664963ULL,
            2588613720361524ULL,
            5802089162353039525ULL,
            214799000387ULL,
            144876322ULL,
            668013605731ULL,
            110616894681956ULL,
            1601657732871812ULL,
            430945547955ULL,
            3156382366321172ULL,
            7644494644932993285ULL,
            3928124806469601813ULL,
            3155990846772900ULL,
            339991010498708ULL,
            10743689387941597493ULL,
            5103845475ULL,
            105070898ULL,
            3928064910068824213ULL,
            156265010ULL,
            1305138421793636ULL,
            27185ULL,
            195459938ULL,
            567044449971ULL,
            382447549283ULL,
            2175279159592324ULL,
            443529919251ULL,
            195059004769796ULL,
            2165424908404116ULL,
            1554158691063110021ULL,
            504228368803ULL,
            1436350466655236ULL,
            27584723588724ULL,
            1900945754488837749ULL,
            122971970ULL,
            443829749251ULL,
            302601798803ULL,
            108558722ULL,
            724700725875ULL,
            43570095105972ULL,
            2295263717447940ULL,
            2860446751369014181ULL,
            2165106202149444ULL,
            69275726195ULL,
            2860543885641537797ULL,
            2165106320445780ULL,
            2280890014640004ULL,
            11820349930268368933ULL,
            8721082628082003989ULL,
            127050770ULL,
            503707084675ULL,
            122834978ULL,
            2538193642857604ULL,
            10129ULL,
            801441490467ULL,
            2923200302876740ULL,
            1443359556281892ULL,
            2901063790822564949ULL,
            2728339631923524ULL,
            7103874718248233397ULL,
            12775311047932294245ULL,
            95520290ULL,
            2623783208098404ULL,
            1900908618382410757ULL,
            137742672547ULL,
            2323440239468964ULL,
            362478212387ULL,
            727199575803140ULL,
            73425410ULL,
            34337ULL,
            163101314ULL,
            668566030659ULL,
            801204361987ULL,
            73030562ULL,
            591509145619ULL,
            162574594ULL,
            100608342969108ULL,
            5553ULL,
            724147968595ULL,
            1436604830452292ULL,
            176259090ULL,
            42001ULL,
            143955266ULL,
            2385ULL,
            18433ULL,
            0ULL,
    };

    const int edges[12][2] = {
            {0, 1},
            {2, 3},
            {4, 5},
            {6, 7},
            {0, 2},
            {1, 3},
            {4, 6},
            {5, 7},
            {0, 4},
            {1, 5},
            {2, 6},
            {3, 7}};

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i >= volume_resolution.x - 2 || j >= volume_resolution.y - 2)
        return;
    float x = voxel_size * i;
    float y = voxel_size * j;
    int resolutionX = volume_resolution.x;
    int resolutionY = volume_resolution.y;
    int resolutionZ = volume_resolution.z;

    for (int k = 0; k < volume_resolution.z - 1; k++) {
        int voxel_index = i + j * resolutionX + k * resolutionX * resolutionY;
        float value = value_volume[78 + 256 * 163 + 256 * 256 * 241];
        float z = voxel_size * k;
        float3 vertices[8] = {
                float3{x, y, z},
                float3{x + voxel_size, y, z},
                float3{x, y + voxel_size, z},
                float3{x + voxel_size, y + voxel_size, z},
                float3{x, y, z + voxel_size},
                float3{x + voxel_size, y, z + voxel_size},
                float3{x, y + voxel_size, z + voxel_size},
                float3{x + voxel_size, y + voxel_size, z + voxel_size}};
        int t = 0;
        float f[8];
        bool flag = true;
        for (int h = 7; h >= 0; h--) {
            int index_x = int(vertices[h].x / voxel_size);
            int index_y = int(vertices[h].y / voxel_size);
            int index_z = int(vertices[h].z / voxel_size);
            int index = index_x + index_y * resolutionX + index_z * resolutionX * resolutionY;
            f[h] = value_volume[index];

            if (f[h] == 0 || fabs(f[h]) > 0.99) {
                flag = false;
                break;
            }
            t = (t << 1) + (f[h] < 0.0f);
        }
        if (!flag || t == 0 || t == 255)
            continue;
        unsigned long long tmp = triangles[t];
        int triangleNum = tmp & 0xF;
        tmp >>= 4;
        for (int h = 0; h < triangleNum * 3; h++) {
            int edgeIndex = tmp & 0xF;
            tmp >>= 4;

            int v1 = edges[edgeIndex][0];
            int v2 = edges[edgeIndex][1];

            float t1 = std::fabs(f[v1]);
            float t2 = std::fabs(f[v2]);
            float sum_x = (t1 * vertices[v1].x + t2 * vertices[v2].x) / (t1 + t2);
            float sum_y = (t1 * vertices[v1].y + t2 * vertices[v2].y) / (t1 + t2);
            float sum_z = (t1 * vertices[v1].z + t2 * vertices[v2].z) / (t1 + t2);
            //            printf("%d, %d, %d, %d\n", i, j, k,voxel_index);
            points[12 * voxel_index + h] = float3{sum_x, sum_y, sum_z};
            indices[12 * voxel_index + h] = -1;
        }
    }
}

void extractMesh(const PtrStep<float> &value_volume, const int3 &volume_resolution,
                 float voxel_size, PtrSz<float3> points, thrustDvec<int> &indices) {
    int voxel_num = volume_resolution.x * volume_resolution.y * volume_resolution.z;
    indices.resize(12 * voxel_num, 0);
    int *indices_ptr = trDptr(indices);
    dim3 block(32, 16);
    dim3 grid(1, 1, 1);
    grid.x = divUp(volume_resolution.x, block.x);
    grid.y = divUp(volume_resolution.y, block.y);

    extractMeshKernal<<<grid, block>>>(value_volume, volume_resolution, voxel_size, points, indices_ptr);
    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaDeviceSynchronize());
}
