//
// Created by MiseTy on 2021/11/2.
//
#include "RayCaster.h"
#include "TsdfFusion.h"

__device__ __forceinline__ float
getMinTime(const float3 &volume_max, const float3 &origin, const float3 &dir) {
    float txmin = ((dir.x > 0 ? 0.f : volume_max.x) - origin.x) / dir.x;
    float tymin = ((dir.y > 0 ? 0.f : volume_max.y) - origin.y) / dir.y;
    float tzmin = ((dir.z > 0 ? 0.f : volume_max.z) - origin.z) / dir.z;

    return fmax(fmax(txmin, tymin), tzmin);
}

__device__ __forceinline__ float
getMaxTime(const float3 &volume_max, const float3 &origin, const float3 &dir) {
    float txmax = ((dir.x > 0 ? volume_max.x : 0.f) - origin.x) / dir.x;
    float tymax = ((dir.y > 0 ? volume_max.y : 0.f) - origin.y) / dir.y;
    float tzmax = ((dir.z > 0 ? volume_max.z : 0.f) - origin.z) / dir.z;

    return fmin(fmin(txmax, tymax), tzmax);
}


struct RayCaster {
    enum { CTA_SIZE_X = 32,
           CTA_SIZE_Y = 8 };

    MatS33 Rc2v;//	camera to volume
    devComplex3 tc2v;

    MatS33 Rv2w;//	volume to world
    devComplex3 tv2w;

    int3 volume_resolution;//	resolution of the volume
    float voxel_size;
    float3 volume_size;

    float time_step;
    int cols, rows;

    PtrStep<float> value_volume;
    PtrStep<int> weight_volume;
    PtrStep<float> grad_volume;

    Intr intr;

    mutable PtrStep<devComplex> nmap;
    mutable PtrStep<devComplex> vmap;

    __device__ __forceinline__ int sgn(float val) const {
        return (0.0f < val) - (val < 0.0f);
    }

    __device__ __forceinline__ devComplex3 get_ray_next(int x, int y) const {
        devComplex3 ray_next;
        ray_next.x = (x - intr.cx) / intr.fx;
        ray_next.y = (y - intr.cy) / intr.fy;
        ray_next.z = 1;
        return ray_next;
    }

    __device__ __forceinline__ bool checkInds(const int3 &g) const {
        return (g.x >= 0 && g.y >= 0 && g.z >= 0 && g.x < volume_resolution.x &&
                g.y < volume_resolution.y && g.z < volume_resolution.z);
    }

    __device__ __forceinline__ devComplex readTsdf(int x, int y, int z) const {
        //	calculate the storage cell
        x = x % volume_resolution.x;
        y = y % volume_resolution.y;
        z = z % volume_resolution.z;
        devComplex res = unpack_tsdf(value_volume.ptr(volume_resolution.y * z + y)[x],
                                     grad_volume.ptr(volume_resolution.y * z + y)[x]);
        res += 1e-5f;
        return res;
    }

    __device__ __forceinline__ int3 getVoxel(float3 point) const {
        int vx = __float2int_rd(point.x / voxel_size);// round to negative infinity
        int vy = __float2int_rd(point.y / voxel_size);
        int vz = __float2int_rd(point.z / voxel_size);

        return make_int3(vx, vy, vz);
    }

    __device__ __forceinline__ devComplex interpolateTrilineary(const devComplex3 &origin,
                                                                const devComplex3 &dir,
                                                                float time) const {
        return interpolateTrilineary(origin + dir * time);
    }
    __device__ __forceinline__ devComplex interpolateTrilineary_debug(const devComplex3 &origin,
                                                                      const devComplex3 &dir,
                                                                      float time) const {
        return interpolateTrilineary_debug(origin + dir * time);
    }

    __device__ __forceinline__ devComplex
    interpolateTrilineary(const devComplex3 &point) const {
        int3 g = getVoxel(real(point));
        float qnan = cx::numeric_limits<float>::quiet_NaN();

        if (g.x <= 0 || g.x >= volume_resolution.x - 1)
            return {qnan, 0};

        if (g.y <= 0 || g.y >= volume_resolution.y - 1)
            return {qnan, 0};

        if (g.z <= 0 || g.z >= volume_resolution.z - 1)
            return {qnan, 0};

        float vx = (g.x + 0.5f) * voxel_size;
        float vy = (g.y + 0.5f) * voxel_size;
        float vz = (g.z + 0.5f) * voxel_size;

        g.x += -(sgn(vx - point.x.real()) + 1) >>
               1;// g.x = (point.x < vx) ? (g.x - 1) : g.x;
        g.y += -(sgn(vy - point.y.real()) + 1) >>
               1;// g.y = (point.y < vy) ? (g.y - 1) : g.y;
        g.z += -(sgn(vz - point.z.real()) + 1) >>
               1;// g.z = (point.z < vz) ? (g.z - 1) : g.z;

        devComplex a0 = (point.x - (g.x + 0.5f) * voxel_size) / voxel_size;
        devComplex b0 = (point.y - (g.y + 0.5f) * voxel_size) / voxel_size;
        devComplex c0 = (point.z - (g.z + 0.5f) * voxel_size) / voxel_size;
        devComplex one(1.0f, 0.0f);
        devComplex a1 = one - a0;
        devComplex b1 = one - b0;
        devComplex c1 = one - c0;

        devComplex res = readTsdf(g.x + 0, g.y + 0, g.z + 0) * a1 * b1 * c1 +
                         readTsdf(g.x + 0, g.y + 0, g.z + 1) * a1 * b1 * c0 +
                         readTsdf(g.x + 0, g.y + 1, g.z + 0) * a1 * b0 * c1 +
                         readTsdf(g.x + 0, g.y + 1, g.z + 1) * a1 * b0 * c0 +
                         readTsdf(g.x + 1, g.y + 0, g.z + 0) * a0 * b1 * c1 +
                         readTsdf(g.x + 1, g.y + 0, g.z + 1) * a0 * b1 * c0 +
                         readTsdf(g.x + 1, g.y + 1, g.z + 0) * a0 * b0 * c1 +
                         readTsdf(g.x + 1, g.y + 1, g.z + 1) * a0 * b0 * c0;
        return res;
    }

    __device__ __forceinline__ devComplex
    interpolateTrilineary_debug(const devComplex3 &point) const {
        int3 g = getVoxel(real(point));
        float qnan = cx::numeric_limits<float>::quiet_NaN();

        if (g.x <= 0 || g.x >= volume_resolution.x - 1)
            return {qnan, 0};

        if (g.y <= 0 || g.y >= volume_resolution.y - 1)
            return {qnan, 0};

        if (g.z <= 0 || g.z >= volume_resolution.z - 1)
            return {qnan, 0};

        float vx = (g.x + 0.5f) * voxel_size;
        float vy = (g.y + 0.5f) * voxel_size;
        float vz = (g.z + 0.5f) * voxel_size;

        g.x += -(sgn(vx - point.x.real()) + 1) >>
               1;// g.x = (point.x < vx) ? (g.x - 1) : g.x;
        g.y += -(sgn(vy - point.y.real()) + 1) >>
               1;// g.y = (point.y < vy) ? (g.y - 1) : g.y;
        g.z += -(sgn(vz - point.z.real()) + 1) >>
               1;// g.z = (point.z < vz) ? (g.z - 1) : g.z;

        devComplex a0 = (point.x - (g.x + 0.5f) * voxel_size) / voxel_size;
        devComplex b0 = (point.y - (g.y + 0.5f) * voxel_size) / voxel_size;
        devComplex c0 = (point.z - (g.z + 0.5f) * voxel_size) / voxel_size;
        devComplex one(1.0f, 0.0f);
        devComplex a1 = one - a0;
        devComplex b1 = one - b0;
        devComplex c1 = one - c0;

        devComplex res = readTsdf(g.x + 0, g.y + 0, g.z + 0) * a1 * b1 * c1 +
                         readTsdf(g.x + 0, g.y + 0, g.z + 1) * a1 * b1 * c0 +
                         readTsdf(g.x + 0, g.y + 1, g.z + 0) * a1 * b0 * c1 +
                         readTsdf(g.x + 0, g.y + 1, g.z + 1) * a1 * b0 * c0 +
                         readTsdf(g.x + 1, g.y + 0, g.z + 0) * a0 * b1 * c1 +
                         readTsdf(g.x + 1, g.y + 0, g.z + 1) * a0 * b1 * c0 +
                         readTsdf(g.x + 1, g.y + 1, g.z + 0) * a0 * b0 * c1 +
                         readTsdf(g.x + 1, g.y + 1, g.z + 1) * a0 * b0 * c0;
        printf("%d, %d, %d\n", g.x, g.y, g.z);
        printf("%f1\n", res.real());

        printf("%f1, %f\n", a0.real(), a0.imag() );
        printf("%f, %f1\n", b0.real(), b0.imag() * 1e10);
        printf("%f, %f1\n", c0.real(), c0.imag() * 1e10);

//        printf("%f, %f1\n", (readTsdf(g.x + 0, g.y + 0, g.z + 0)).real(), (readTsdf(g.x + 0, g.y + 0, g.z + 0)).imag() * 1e10);
//        printf("%f, %f1\n", (readTsdf(g.x + 0, g.y + 0, g.z + 0) * a1).real(), (readTsdf(g.x + 0, g.y + 0, g.z + 0) * a1).imag() * 1e10);
//        printf("%f, %f1\n", (readTsdf(g.x + 0, g.y + 0, g.z + 0) * a1 * b1 * c1).real(), (readTsdf(g.x + 0, g.y + 0, g.z + 0) * a1 * b1 * c1).imag() * 1e10);

        return res;
    }
    __device__ __forceinline__ void operator()() const {
        int x = threadIdx.x + blockIdx.x * CTA_SIZE_X;
        int y = threadIdx.y + blockIdx.y * CTA_SIZE_Y;

        if (x >= cols || y >= rows)
            return;

        vmap.ptr(y)[x] = cx::numeric_limits<float>::quiet_NaN();
        nmap.ptr(y)[x] = cx::numeric_limits<float>::quiet_NaN();

        devComplex3 ray_start = tc2v;
        devComplex3 ray_next = Rc2v * get_ray_next(x, y) + tc2v;
        devComplex3 ray_dir = normalized(ray_next - ray_start);
        // ensure that it isn't a degenerate case
        ray_dir.x = (ray_dir.x == 0.f) ? 1e-15 : ray_dir.x;
        ray_dir.y = (ray_dir.y == 0.f) ? 1e-15 : ray_dir.y;
        ray_dir.z = (ray_dir.z == 0.f) ? 1e-15 : ray_dir.z;

//        // computer time when entry and exit volume
//        float time_start_volume = getMinTime(volume_size, real(ray_start), real(ray_dir));
//        float time_exit_volume = getMaxTime(volume_size, real(ray_start), real(ray_dir));
//        const float min_dist = 0.f1;// in meters
//        time_start_volume = fmax(time_start_volume, min_dist);
//        if (time_start_volume >= time_exit_volume)
//            return;
        float time_start_volume = 0.2;
        float time_exit_volume = 5.0;


        float time_curr = time_start_volume;
        int3 g = getVoxel(real(ray_start + ray_dir * time_curr));
        g.x = max(0, min(g.x, volume_resolution.x - 1));
        g.y = max(0, min(g.y, volume_resolution.y - 1));
        g.z = max(0, min(g.z, volume_resolution.z - 1));
        devComplex tsdf = readTsdf(g.x, g.y, g.z);

        // infinite loop guard
        const float max_time = time_exit_volume;

        for (; time_curr < max_time; time_curr += time_step) {
            devComplex tsdf_prev = tsdf;
            float3 curr_point = real(ray_start + ray_dir * (time_curr + time_step));
            g = getVoxel(curr_point);
            if (!checkInds(g))
                break;

            tsdf = readTsdf(g.x, g.y, g.z);

            if (tsdf_prev.real() < 0.f && tsdf.real() > 0.f)
                break;
            if (tsdf_prev.real() > 0.f && tsdf.real() < 0.f)// zero crossing
            {
                devComplex Ftdt =
                        interpolateTrilineary(ray_start, ray_dir, time_curr + time_step);
                if (isnan(Ftdt.real()))
                    break;
                if(Ftdt.real() ==0)
                    printf("(%d, %d)\n", x, y);
                devComplex Ft = interpolateTrilineary(ray_start, ray_dir, time_curr);
                if (isnan(Ft.real()))
                    break;
                devComplex coef = Ft / (Ftdt - Ft);
                if (Ft.real() < 0.0f || Ftdt.real() > 0.0f)
                    break;
                devComplex Ts = time_curr - time_step * coef;

                devComplex3 vertex_found = ray_start + ray_dir * Ts;
                devComplex3 vertex_found_w = Rv2w * vertex_found + tv2w;
                vmap.ptr(y)[x] = vertex_found_w.x;
                vmap.ptr(y + rows)[x] = vertex_found_w.y;
                vmap.ptr(y + 2 * rows)[x] = vertex_found_w.z;

                g = getVoxel(real(vertex_found));
                if (g.x > 1 && g.y > 1 && g.z > 1 && g.x < volume_resolution.x - 2 &&
                    g.y < volume_resolution.y - 2 && g.z < volume_resolution.z - 2) {
                    devComplex3 t;
                    devComplex3 n;
                    float half_voxel_size = voxel_size * 0.5f;
                    t = vertex_found;
                    t.x += half_voxel_size;
                    devComplex Fx1 = interpolateTrilineary(t);
                    t = vertex_found;
                    t.x -= half_voxel_size;
                    devComplex Fx2 = interpolateTrilineary(t);
                    n.x = (Fx1 - Fx2);

                    t = vertex_found;
                    t.y += half_voxel_size;
                    devComplex Fy1 = interpolateTrilineary(t);
                    t = vertex_found;
                    t.y -= half_voxel_size;
                    devComplex Fy2 = interpolateTrilineary(t);
                    n.y = (Fy1 - Fy2);

                    t = vertex_found;
                    t.z += half_voxel_size;
                    devComplex Fz1 = interpolateTrilineary(t);
                    t = vertex_found;
                    t.z -= half_voxel_size;
                    devComplex Fz2 = interpolateTrilineary(t);
                    n.z = (Fz1 - Fz2);

                    if (squarednorm(n).real() == 0)
                        break;
                    devComplex3 n_g = Rv2w * normalized(n);
                    nmap.ptr(y)[x] = n_g.x;
                    nmap.ptr(y + rows)[x] = n_g.y;
                    nmap.ptr(y + 2 * rows)[x] = n_g.z;
                }
                break;
            }

        } /* for(;;)  */
    }
    __device__ __forceinline__ void debug() const {
        int x = threadIdx.x + blockIdx.x * CTA_SIZE_X;
        int y = threadIdx.y + blockIdx.y * CTA_SIZE_Y;
        if (x >= volume_resolution.x - 1 || y >= volume_resolution.y - 1)
            return;
        for (int z = 0; z < volume_resolution.z - 1; z++) {
            devComplex res = readTsdf(x, y, z);
            printf("%f, %f1\n", res.real(), res.imag());
        }
    }
};


__global__ void rayCastKernel(const RayCaster rc) { rc(); }


void raycast(const Intr &intr, const MatS33 &Rc2v, const devComplex3 &tc2v,
             const MatS33 &Rv2w, const devComplex3 &tv2w, float tranc_dist,
             const int3 &volume_resolution, float voxel_size,
             const PtrStep<float> &value_volume, const PtrStep<float> &grad_volume,
             MapArr &vmap, MapArr &nmap) {

    RayCaster rc;
    rc.Rc2v = Rc2v;
    rc.tc2v = tc2v;
    rc.Rv2w = Rv2w;
    rc.tv2w = tv2w;

    rc.volume_resolution.x = volume_resolution.x;
    rc.volume_resolution.y = volume_resolution.y;
    rc.volume_resolution.z = volume_resolution.z;

    rc.voxel_size = voxel_size;

    rc.volume_size.x = volume_resolution.x * voxel_size;
    rc.volume_size.y = volume_resolution.y * voxel_size;
    rc.volume_size.z = volume_resolution.z * voxel_size;


    rc.time_step = tranc_dist * 0.8f;
    rc.cols = vmap.cols();
    rc.rows = vmap.rows() / 3;

    rc.intr = intr;

    rc.value_volume = value_volume;
    rc.grad_volume = grad_volume;

    rc.vmap = vmap;
    rc.nmap = nmap;

    dim3 block(RayCaster::CTA_SIZE_X, RayCaster::CTA_SIZE_Y);
    dim3 grid(divUp(rc.cols, block.x), divUp(rc.rows, block.y));

    rayCastKernel<<<grid, block>>>(rc);
    cudaSafeCall(cudaGetLastError());
    // cudaSafeCall(cudaDeviceSynchronize());
}