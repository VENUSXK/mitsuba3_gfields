#pragma once

#include <math.h>
#include <mitsuba/render/optix/common.h>
#include <mitsuba/render/optix/math.cuh>

/// Mirror of the transmittance-related \ref RayFlags bits (interaction.h).
enum MiEllipsoidsRayFlags : unsigned int {
    MiStochasticEllipsoids    = 0x1000,
    MiTransmittanceEllipsoids = 0x2000,
    MiSampleEllipsoids        = 0x4000,
    MiCollectEllipsoids       = 0x8000,
    // !!! KEEP IN SYNC WITH interaction.h !!!
};

struct OptixEllipsoidsData {
    float* extents;
    float* data;
    /// Per-primitive opacities/omegas for any-hit transmittance queries
    /// (may be nullptr when the shape carries no such attribute).
    float* opacities;
    float* omegas;
    /// Triangles per shell instance (EllipsoidsMesh); 1 for analytic shells.
    unsigned int subprimitive_count;
    // group consecutive gaussian supports under one acceleration primitive.
    unsigned int primitives_per_column = 1;
    unsigned int primitive_count = 0;
    bool column_mesh = false;

#ifdef __CUDACC__
    DEVICE Vector3f center(unsigned int index) const {
        return Vector3f::load(&data[index * 10]);
    }

    DEVICE Vector3f scale(unsigned int index) const {
        return Vector3f::load(&data[index * 10 + 3]);
    }

    DEVICE Matrix3f rotation(unsigned int index) const {
        auto q = Vector4f::load(&data[index * 10 + 6]);
        q = q * SqrtTwo;
        float xx = q.x() * q.x(), yy = q.y() * q.y(), zz = q.z() * q.z(),
              xy = q.x() * q.y(), xz = q.x() * q.z(), yz = q.y() * q.z(),
              xw = q.x() * q.w(), yw = q.y() * q.w(), zw = q.z() * q.w();
        auto m = Matrix3f(); // Already transposed
        m[0] = Vector3f(1.f - (yy + zz), xy - zw, xz + yw);
        m[1] = Vector3f(xy + zw, 1.f - (xx + zz), yz - xw);
        m[2] = Vector3f(xz - yw,  yz + xw, 1.f - (xx + yy));
        return m;
    }
#endif
};

#ifdef __CUDACC__
// reuse the same support test for grouped traversal and per-gaussian hits.
DEVICE bool ellipsoid_support_hit(const OptixEllipsoidsData &ellipsoid, unsigned int prim_index,
                                  const Ray3f &ray, bool include_interior, float &t, unsigned int &hit_kind) {
    Vector3f center = ellipsoid.center(prim_index);
    Vector3f scale = ellipsoid.scale(prim_index) * ellipsoid.extents[prim_index];
    Matrix3f rotation = ellipsoid.rotation(prim_index);
    Vector3f o = rotation.transposed_prod(ray.o - center) / scale;
    Vector3f d = rotation.transposed_prod(ray.d) / scale;

    // project onto the closest plane without magnifying small-scale rays.
    float plane_t = dot(-o, d) / squared_norm(d);
    Vector3f plane_p = o + d * plane_t;
    float near_t, far_t;
    bool solution_found = solve_quadratic(squared_norm(d), 2.f * dot(plane_p, d), squared_norm(plane_p) - 1.f, near_t, far_t);
    near_t += plane_t;
    far_t += plane_t;

    // expose containing supports only to coverage and sampling queries.
    if (include_interior && solution_found && near_t < 0.f && far_t >= 0.f) {
        t = fmaxf(fminf(far_t, ray.maxt * (1.f - 1e-6f)), 0.f);
        hit_kind = OPTIX_HIT_KIND_TRIANGLE_BACK_FACE;
        return true;
    }
    t = near_t;
    hit_kind = OPTIX_HIT_KIND_TRIANGLE_FRONT_FACE;
    return solution_found && near_t >= 0.f && near_t <= ray.maxt && far_t >= 0.f;
}

extern "C" __global__ void __intersection__ellipsoids() {
    const OptixHitGroupData *sbt_data = (OptixHitGroupData *) optixGetSbtDataPointer();
    const OptixEllipsoidsData &ellipsoid = *((const OptixEllipsoidsData *) sbt_data->data);
    Ray3f ray = get_ray();
    unsigned int count = ellipsoid.primitives_per_column;
    unsigned int first = optixGetPrimitiveIndex() * count;
    bool include_interior = (optixGetRayFlags() & OPTIX_RAY_FLAG_ENFORCE_ANYHIT) != 0;
    bool grouped_anyhit = count > 1 && !(optixGetRayFlags() & OPTIX_RAY_FLAG_DISABLE_ANYHIT);

    // report one group event for any-hit and individual ids for closest-hit.
    for (unsigned int k = 0; k < count; ++k) {
        unsigned int prim_index = first + k;
        float t;
        unsigned int hit_kind;
        if (ellipsoid_support_hit(ellipsoid, prim_index, ray, include_interior, t, hit_kind)) {
            optixReportIntersection(t, hit_kind, prim_index, __float_as_uint(ray.maxt));
            if (grouped_anyhit)
                return;
        }
    }
}

// -----------------------------------------------------------------------
//  Any-hit transmittance machinery (see Scene::ray_transmittance_ellipsoids)
// -----------------------------------------------------------------------

/// PCG-style integer hash -> uniform float in [0, 1)
DEVICE float pcg_uniform(unsigned int v) {
    unsigned int state = v * 747796405u + 2891336453u;
    unsigned int word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    word = (word >> 22u) ^ word;
    return (float) word * (1.0f / 4294967296.0f);
}

/**
 * Analytic full-line signed optical depth of one Gaussian/Gabor primitive
 * along the current world ray. Matches the full-range branch of the Python
 * kernels (gfields.integrators.common.density_integral):
 *
 *     tau = opacity * J * exp(-0.5 (C - B^2 + w^2)) * cos(D - w B)
 *           / (2 pi sx sy sz)
 *
 * with the ray reduced to standardized coordinates (J the unit-speed
 * Jacobian). Pure Gaussians are the omega = 0 special case (cos -> 1).
 */
DEVICE float ellipsoid_fullline_optical_depth(const OptixEllipsoidsData &e,
                                              unsigned int prim_index, bool resolved_index = false) {
    Vector3f center   = e.center(prim_index);
    Vector3f scale    = e.scale(prim_index);
    Matrix3f rotation = e.rotation(prim_index);

    float3 ro = optixGetWorldRayOrigin();
    float3 rd = optixGetWorldRayDirection();
    Vector3f o(ro.x, ro.y, ro.z), d(rd.x, rd.y, rd.z);

    Vector3f w = rotation.transposed_prod(d) / scale;
    Vector3f p = rotation.transposed_prod(o - center) / scale;

    float J = 1.f / norm(w);
    w = w * J;

    float B = dot(w, p);
    float C = dot(p, p);

    float om    = e.omegas ? e.omegas[prim_index] : 0.f;
    float omega = om * (w.x() + w.y() + w.z());
    float D     = om * (p.x() + p.y() + p.z());

    float opacity = e.opacities ? e.opacities[prim_index] : 1.f;

    return opacity * J * expf(-0.5f * (C - B * B + omega * omega))
                   * cosf(D - omega * B)
                   / (TwoPi * scale.x() * scale.y() * scale.z());
}

/**
 * Any-hit handler for ellipsoid primitives (called from
 * __anyhit__transmittance in optix_rt.cu). Two modes, selected by the
 * Mitsuba ray-flag bits carried in payload 0:
 *
 *  - TransmittanceEllipsoids: add the primitive's analytic signed optical
 *    depth to the float accumulator in payload 2, then ignore the hit
 *    (deterministic, order-independent: one traversal returns sum tau_i).
 *
 *  - StochasticEllipsoids: Bernoulli-test the primitive with survival
 *    probability min(1, exp(-tau)); the RNG is a stateless hash of the
 *    per-ray seed (payload 1) and the primitive index, so each primitive is
 *    tested independently and traversal order does not matter. Accepting
 *    the hit (absorption) terminates traversal via
 *    OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT at the trace call.
 *
 * The shell mesh is back-face culled, so each primitive triggers exactly one
 * any-hit invocation (its front-face entry) per traversal. Rays that start
 * inside a shell miss that primitive, mirroring the walk-based integrators;
 * callers handle vertex-enclosing primitives analytically.
 */
DEVICE bool transmittance_anyhit_ellipsoids(void* data,
                                            unsigned int prim_index,
                                            unsigned int mi_ray_flags, bool resolved_index = false) {
    const OptixEllipsoidsData &e = *((const OptixEllipsoidsData *) data);

    // EllipsoidsMesh reports triangle indices; map to the shell instance.
    if (!resolved_index && e.subprimitive_count > 1)
        prim_index /= e.subprimitive_count;

    float tau = ellipsoid_fullline_optical_depth(e, prim_index);

    if (mi_ray_flags & MiTransmittanceEllipsoids) {
        float acc = __uint_as_float(optixGetPayload_2());
        optixSetPayload_2(__float_as_uint(acc + tau));
        return false;
    }

    // Stochastic Bernoulli test. Net-amplifying signed primitives (tau < 0)
    // survive with probability one (their exp(-tau) > 1 residual is the
    // callers' residual_weight_clamp convention).
    float p_survive = expf(-fmaxf(tau, 0.f));
    unsigned int seed = optixGetPayload_1();
    float u = pcg_uniform(seed ^ (prim_index * 0x9E3779B9u));
    // let the outer handler ignore once after every child has contributed.
    return u >= p_survive;
}

/**
 * Any-hit handler for the SampleEllipsoids distance-sampling queries
 * (Scene::ray_sample_ellipsoids). Each intersected primitive draws its
 * closed-form inverse-CDF free flight over the kernel's full unbounded
 * support (the extent shell only serves as the BVH proxy that discovers
 * the primitive; its clipped tail mass is negligible at >= 3 sigma) with
 * a stateless per-primitive RNG, and min-folds the result: the minimum
 * over independent per-primitive flights is exactly one free flight of
 * the (envelope) mixture, so traversal order does not matter. Payload
 * layout:
 *
 *   payload 0 (in)     Mitsuba ray flags
 *   payload 1 (in)     per-ray RNG seed
 *   payload 2 (in/out) float bits of the running minimum flight (world t
 *                      from the ray origin; +inf when no collision)
 *   payload 3 (out)    shape registry id of the winning primitive
 *   payload 4 (out)    primitive index of the winning primitive
 *   payload 5 (in)     float bits of the restart distance t_start (world
 *                      units): flights are drawn conditioned on survival
 *                      past it (0 for a fresh query)
 *
 * For Gabor primitives the flight is drawn from the *Gaussian envelope*
 * (the |cos| <= 1 modulation is majorized away); the caller runs the
 * decomposition-tracking acceptance test at the returned tentative
 * collision. For pure Gaussians the envelope is the kernel itself and the
 * sample is exact. The query traces with ENFORCE_ANYHIT and no back-face
 * culling, so primitives containing the ray origin are visible too
 * (interior-hit reporting for analytic shells, back faces for mesh
 * shells); every flight starts at max(t_start, 0) and integrates the
 * remaining unbounded mass. Mesh shells ahead of the origin fire twice
 * (front + back); both invocations compute the identical flight from the
 * stateless per-primitive hash, so the min-fold is idempotent (the
 * duplicate is skipped early as an optimization).
 */
DEVICE void sample_anyhit_ellipsoids(const OptixHitGroupData *sbt_data,
                                     unsigned int prim_index, bool backfacing, bool resolved_index = false) {
    const OptixEllipsoidsData &e = *((const OptixEllipsoidsData *) sbt_data->data);
    if (!resolved_index && e.subprimitive_count > 1)
        prim_index /= e.subprimitive_count;

    Vector3f center   = e.center(prim_index);
    Vector3f scale    = e.scale(prim_index);
    Matrix3f rotation = e.rotation(prim_index);

    float3 ro = optixGetWorldRayOrigin();
    float3 rd = optixGetWorldRayDirection();
    Vector3f o(ro.x, ro.y, ro.z), d(rd.x, rd.y, rd.z);

    // Standardized (unit-covariance) frame anchored at the ray origin.
    Vector3f w = rotation.transposed_prod(d) / scale;
    Vector3f p = rotation.transposed_prod(o - center) / scale;
    float J = 1.f / norm(w);
    w = w * J;
    float B = dot(w, p);
    float C = dot(p, p);

    // Mesh shells fire twice without back-face culling (front entry + back
    // exit). The back-face invocation is only meaningful when the ray
    // starts inside this primitive's shell (no front face fires then);
    // otherwise it duplicates the front-face one -- skip before the
    // transcendentals. The shell roots are used for nothing else: flights
    // integrate the kernel's full unbounded support.
    if (backfacing) {
        float ext  = e.extents[prim_index];
        float disc = B * B - C + ext * ext;
        float t_near = -B - sqrtf(fmaxf(disc, 0.f));
        if (!(t_near < 0.f)) {
            return;
        }
    }

    // Restart support (decomposition-tracking rounds): condition the
    // flight on survival past t_start. World t = J * standardized t.
    float t_start = __uint_as_float(optixGetPayload_5());
    float t0 = fmaxf(t_start / J, 0.f);

    // Envelope amplitude in density_integral's normalization (the full
    // standardized-line optical depth is A/2 * (erf((t+B)/sqrt2) - ...)).
    float opacity = e.opacities ? e.opacities[prim_index] : 1.f;
    float A = opacity * J * expf(-0.5f * (C - B * B))
                      / (TwoPi * scale.x() * scale.y() * scale.z());
    if (!(A > 0.f) || !isfinite(A)) {
        return;
    }

    unsigned int seed = optixGetPayload_1();
    float u   = pcg_uniform(seed ^ (prim_index * 0x9E3779B9u));
    float chi = -logf(fmaxf(1.f - u, 1e-38f));

    const float SqrtHalf = 0.7071067811865476f;
    float arg = erff(SqrtHalf * (t0 + B)) + 2.f * chi / A;
    if (!(arg < 1.f)) { // remaining envelope mass exhausted
        return;
    }
    float th = sqrtf(2.f) * erfinvf(arg) - B;

    float tw = J * th;
    if (tw < __uint_as_float(optixGetPayload_2())) {
        optixSetPayload_2(__float_as_uint(tw));
        optixSetPayload_3(sbt_data->shape_registry_id);
        optixSetPayload_4(prim_index);
    }
}

/**
 * Any-hit handler for the CollectEllipsoids coverage queries
 * (Scene::ray_collect_ellipsoids): append the primitives whose extent
 * shell contains the ray ORIGIN to a caller-provided per-ray device
 * buffer. Containing primitives are visible to the traversal through
 * their forward exit hit (interior reporting in __intersection__ellipsoids
 * for analytic shells, un-culled back faces for mesh shells: both fire
 * exactly once per primitive on convex shells); non-containing primitives
 * ahead of the origin also fire but fail the containment test. Payloads:
 *
 *   payload 0 (in)     Mitsuba ray flags
 *   payload 1 (in/out) number of primitives written
 *   payload 2 (in)     low 32 bits of this ray's buffer slot address
 *   payload 3 (in)     high 32 bits of this ray's buffer slot address
 *   payload 4 (in)     slot capacity (pairs); extras are dropped like a
 *                      full coverage stack in the walk-based integrator
 *
 * Each entry is an (shape_registry_id, prim_index) pair of unsigned ints.
 */
DEVICE void collect_anyhit_ellipsoids(const OptixHitGroupData *sbt_data,
                                      unsigned int prim_index, bool resolved_index = false) {
    const OptixEllipsoidsData &e = *((const OptixEllipsoidsData *) sbt_data->data);
    if (!resolved_index && e.subprimitive_count > 1)
        prim_index /= e.subprimitive_count;

    Vector3f center   = e.center(prim_index);
    Vector3f scale    = e.scale(prim_index);
    Matrix3f rotation = e.rotation(prim_index);
    float ext = e.extents[prim_index];

    float3 ro = optixGetWorldRayOrigin();
    Vector3f x(ro.x, ro.y, ro.z);
    Vector3f q = rotation.transposed_prod(x - center) / (scale * ext);
    float r2 = squared_norm(q);

    if (isfinite(r2) && r2 <= 1.f) {
        unsigned int count = optixGetPayload_1();
        unsigned int cap   = optixGetPayload_4();
        if (count < cap) {
            unsigned long long ptr =
                ((unsigned long long) optixGetPayload_3() << 32) |
                 (unsigned long long) optixGetPayload_2();
            unsigned int *out = (unsigned int *) ptr;
            out[2 * count]     = sbt_data->shape_registry_id;
            out[2 * count + 1] = prim_index;
            optixSetPayload_1(count + 1);
        }
    }
}
#endif
