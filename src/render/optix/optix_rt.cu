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

    // ordinary traces ignore proxies; the dedicated pass visits complete shells.
    const OptixEllipsoidsData *ellipsoids = is_ellipsoids ? (const OptixEllipsoidsData *) sbt_data->data : nullptr;
    bool column_mesh = ellipsoids && shape_type == (unsigned int) ShapeType::EllipsoidsMesh && ellipsoids->column_mesh;
    bool column_pass = (mi_ray_flags & MiColumnMeshPass) != 0;
    if (column_pass && !column_mesh) {
        optixIgnoreIntersection();
        return;
    }
    if (column_mesh && !column_pass) {
        optixIgnoreIntersection();
        return;
    }
    if (!mi_ray_flags)
        return;

    // use only the exit face of each closed box so containing shells fire once.
    if (column_mesh) {
        if (!optixIsBackFaceHit()) {
            optixIgnoreIntersection();
            return;
        }
        const auto &e = *ellipsoids;
        unsigned int first = (prim_index / e.subprimitive_count) * e.primitives_per_column;
        Ray3f ray = get_ray();
        bool closest = (mi_ray_flags & MiColumnMeshClosest) != 0;
        if (closest)
            ray.maxt = __uint_as_float(optixGetPayload_1());
        else if (!(mi_ray_flags & (MiSampleEllipsoids | MiCollectEllipsoids)))
            ray.maxt = __uint_as_float(optixGetPayload_3());

        // resolve exact child ids before sharing the existing integration helpers.
        for (unsigned int k = 0; k < e.primitives_per_column; ++k) {
            unsigned int child = first + k, kind;
            float t;
            bool interior = !closest && (mi_ray_flags & (MiSampleEllipsoids | MiCollectEllipsoids));
            if (!ellipsoid_support_hit(e, child, ray, interior, t, kind))
                continue;
            if (closest) {
                float best = __uint_as_float(optixGetPayload_2());
                unsigned int best_shape = optixGetPayload_3(), best_child = optixGetPayload_4();
                if (t < best || (t == best && (sbt_data->shape_registry_id < best_shape || (sbt_data->shape_registry_id == best_shape && child < best_child)))) {
                    optixSetPayload_2(__float_as_uint(t));
                    optixSetPayload_3(sbt_data->shape_registry_id);
                    optixSetPayload_4(child);
                    optixSetPayload_5(optixGetInstanceId());
                }
            } else if (mi_ray_flags & MiSampleEllipsoids)
                sample_anyhit_ellipsoids(sbt_data, child, kind == OPTIX_HIT_KIND_TRIANGLE_BACK_FACE, true);
            else if (mi_ray_flags & MiCollectEllipsoids)
                collect_anyhit_ellipsoids(sbt_data, child, true);
            else if (transmittance_anyhit_ellipsoids(sbt_data->data, child, mi_ray_flags, true))
                return;
        }
        optixIgnoreIntersection();
        return;
    }

    if (is_ellipsoids) {
        const OptixEllipsoidsData &e = *((const OptixEllipsoidsData *) sbt_data->data);
        bool grouped = shape_type == (unsigned int) ShapeType::Ellipsoids && e.primitives_per_column > 1;
        unsigned int count = grouped ? e.primitives_per_column : 1;
        unsigned int first = grouped ? prim_index * count : prim_index;
        Ray3f ray = get_ray();
        if (grouped)
            ray.maxt = __uint_as_float(optixGetAttribute_1());

        // test child supports and reuse the existing per-gaussian contributions.
        for (unsigned int k = 0; k < count; ++k) {
            unsigned int child_index = first + k;
            float t;
            unsigned int hit_kind;
            bool interior = (optixGetRayFlags() & OPTIX_RAY_FLAG_ENFORCE_ANYHIT) != 0;
            if (grouped && !ellipsoid_support_hit(e, child_index, ray, interior, t, hit_kind))
                continue;

            // accumulate every child before ignoring the single column event.
            bool backfacing = grouped ? hit_kind == OPTIX_HIT_KIND_TRIANGLE_BACK_FACE : optixIsBackFaceHit();
            if (mi_ray_flags & MiSampleEllipsoids)
                sample_anyhit_ellipsoids(sbt_data, child_index, backfacing);
            else if (mi_ray_flags & MiCollectEllipsoids)
                collect_anyhit_ellipsoids(sbt_data, child_index);
            else if (transmittance_anyhit_ellipsoids(sbt_data->data, child_index, mi_ray_flags))
                return;
        }
        optixIgnoreIntersection();
    } else if (mi_ray_flags & MiCollectEllipsoids) {
        optixIgnoreIntersection();
    }
    // else: accept the hit (opaque blocker; in sample mode the committed
    // closest surface clips the query and is returned as the preliminary
    // intersection)
}
