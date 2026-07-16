/*
    accel_optix.h -- OptiX acceleration backend declarations.
*/

#pragma once

#if defined(MI_ENABLE_CUDA)

#include <mitsuba/render/fwd.h>
#include <drjit/array_traverse.h>
#include <tuple>

NAMESPACE_BEGIN(mitsuba)

/// Per-scene OptiX state (SBT/GAS/IAS), defined in src/render/scene_optix.inl
struct MiOptixSceneState;

/// Vectorized GPU ray tracing acceleration via CUDA/OptiX
template <typename Float, typename Spectrum>
struct OptixAccel {
    MI_IMPORT_TYPES(Shape, ShapePtr)
    OptixAccel() = default;
    DRJIT_NON_COPYABLE(OptixAccel)

    ~OptixAccel() { release(); }

    // --- Lifecycle (bodies in scene_optix.inl) ---
    void init(Scene<Float, Spectrum> *scene, const Properties &props);
    void rebuild(Scene<Float, Spectrum> *scene);
    void release();

    static void static_initialization() { }
    static void static_shutdown();

    // --- Ray queries (bodies in scene_optix.inl) ---
    PreliminaryIntersection3f ray_intersect_preliminary(
        const Scene<Float, Spectrum> *scene, const Ray3f &ray, Mask coherent,
        bool reorder, UInt32 reorder_hint, uint32_t reorder_hint_bits,
        Mask active) const;
    Mask ray_test(const Scene<Float, Spectrum> *scene, const Ray3f &ray,
                  Mask coherent, Mask active) const;
    /// Any-hit-driven ellipsoid transmittance query
    /// (see \ref Scene::ray_transmittance_ellipsoids).
    Float ray_transmittance_ellipsoids(const Scene<Float, Spectrum> *scene,
                                       const Ray3f &ray, const UInt32 &seed,
                                       uint32_t mode_flags, Mask active) const;
    /// Any-hit-driven ellipsoid free-flight distance sampling
    /// (see \ref Scene::ray_sample_ellipsoids). Returns the sampled flight
    /// (world t; +inf when no collision before the closest blocking
    /// surface), the winning primitive's shape registry id and primitive
    /// index, and the blocking surface as a preliminary intersection.
    std::tuple<Float, UInt32, UInt32, PreliminaryIntersection3f>
    ray_sample_ellipsoids(const Scene<Float, Spectrum> *scene,
                          const Ray3f &ray, const UInt32 &seed,
                          const Float &t_start, Mask active) const;
    /// Any-hit-driven collection of the ellipsoid primitives containing the
    /// ray origin (see \ref Scene::ray_collect_ellipsoids). Writes up to
    /// `capacity` (shape_registry_id, prim_index) pairs per ray at the
    /// per-lane device address `slot_addr` and returns the per-ray count.
    UInt32 ray_collect_ellipsoids(const Scene<Float, Spectrum> *scene,
                                  const Ray3f &ray, const UInt64 &slot_addr,
                                  uint32_t capacity, Mask active) const;
    /// OptiX exposes no brute-force traversal, so this throws.
    SurfaceInteraction3f ray_intersect_naive(
        const Scene<Float, Spectrum> *scene, const Ray3f &ray,
        Mask active) const;

    // --- Declarative traversal (the IAS handle + the rebindable SBT handle) ---
    DRJIT_TRAVERSE(OptixAccel, accel_handle, sbt_handle)

    /// Heap-allocated native OptiX state.
    MiOptixSceneState *state = nullptr;
    /// Freeze-visible IAS and SBT owner handles.
    UInt64 accel_handle;
    UInt64 sbt_handle;
};

NAMESPACE_END(mitsuba)

#endif // MI_ENABLE_CUDA
