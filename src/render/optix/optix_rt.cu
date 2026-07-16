#include <math_constants.h>

#include <mitsuba/render/optix/matrix.cuh>
#include <mitsuba/render/optix/common.h>

// Include all shapes CUDA headers to generate their PTX programs
#include <mitsuba/render/optix/shapes.h>

enum class ShapeType : uint32_t {
    Mesh            = 1u << 0,
    Rectangle       = Mesh | (1u << 1),
    BSplineCurve    = 1u << 2,
    LinearCurve     = 1u << 3,
    Cylinder        = 1u << 4,
    Disk            = 1u << 5,
    SDFGrid         = 1u << 6,
    Sphere          = 1u << 7,
    Ellipsoids      = 1u << 8,
    EllipsoidsMesh  = Mesh | (1u << 9),
    Instance        = 1u << 10,
    ShapeGroup      = 1u << 11,
    // IMPORTANT: Keep in sync with shape.h!
};

/**
 * Any-hit program for the volumetric-primitive queries issued by
 * \c Scene::ray_transmittance_ellipsoids, \c Scene::ray_sample_ellipsoids
 * and \c Scene::ray_collect_ellipsoids (see interaction.h RayFlags and
 * shapes/optix/ellipsoids.cuh for the per-mode payload layouts; payload 0
 * always carries the Mitsuba ray-flag bits selecting the mode).
 *
 * Ellipsoid primitives dispatch to the shared handlers. Hits on any other
 * shape are accepted as-is so ordinary geometry blocks/clips the query --
 * except in collect mode, where surfaces are irrelevant to point coverage
 * and committing one would prune containing shells behind it.
 * Regular ray_intersect / ray_test traces disable any-hit execution
 * entirely (OPTIX_RAY_FLAG_DISABLE_ANYHIT + invoke=false), so this program
 * only runs for the dedicated queries.
 */
extern "C" __global__ void __anyhit__transmittance() {
    const OptixHitGroupData *sbt_data =
        (OptixHitGroupData *) optixGetSbtDataPointer();
    unsigned int shape_type = sbt_data->shape_type;
    unsigned int prim_index = optixGetPrimitiveIndex();
    unsigned int mi_ray_flags = optixGetPayload_0();

    bool is_ellipsoids =
        shape_type == (unsigned int) ShapeType::EllipsoidsMesh ||
        shape_type == (unsigned int) ShapeType::Ellipsoids;

    if (is_ellipsoids) {
        if (mi_ray_flags & MiSampleEllipsoids)
            sample_anyhit_ellipsoids(sbt_data, prim_index);
        else if (mi_ray_flags & MiCollectEllipsoids)
            collect_anyhit_ellipsoids(sbt_data, prim_index);
        else
            transmittance_anyhit_ellipsoids(sbt_data->data, prim_index,
                                            mi_ray_flags);
    } else if (mi_ray_flags & MiCollectEllipsoids) {
        optixIgnoreIntersection();
    }
    // else: accept the hit (opaque blocker; in sample mode the committed
    // closest surface clips the query and is returned as the preliminary
    // intersection)
}
