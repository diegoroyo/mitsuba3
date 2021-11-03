#include <mitsuba/render/volume.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/render/interaction.h>

NAMESPACE_BEGIN(mitsuba)

// =======================================================================
//! @{ \name Volume implementations
// =======================================================================

MTS_VARIANT Volume<Float, Spectrum>::Volume(const Properties &props) {
    m_to_local = props.get<ScalarTransform4f>("to_world", ScalarTransform4f()).inverse();
    update_bbox();
}

MTS_VARIANT typename Volume<Float, Spectrum>::UnpolarizedSpectrum
Volume<Float, Spectrum>::eval(const Interaction3f &, Mask) const {
    NotImplementedError("eval");
}

MTS_VARIANT Float Volume<Float, Spectrum>::eval_1(const Interaction3f &, Mask) const {
    NotImplementedError("eval_1");
}

MTS_VARIANT typename Volume<Float, Spectrum>::Vector3f
Volume<Float, Spectrum>::eval_3(const Interaction3f &, Mask) const {
    NotImplementedError("eval_3");
}

MTS_VARIANT ek::Array<Float, 6>
Volume<Float, Spectrum>::eval_6(const Interaction3f &, Mask) const {
    NotImplementedError("eval_6");
}

MTS_VARIANT std::pair<typename Volume<Float, Spectrum>::UnpolarizedSpectrum,
                      typename Volume<Float, Spectrum>::Vector3f>
Volume<Float, Spectrum>::eval_gradient(const Interaction3f & /*it*/, Mask /*active*/) const {
    NotImplementedError("eval_gradient");
}

MTS_VARIANT typename Volume<Float, Spectrum>::ScalarFloat
Volume<Float, Spectrum>::max() const { NotImplementedError("max"); }

MTS_VARIANT typename Volume<Float, Spectrum>::TensorXf
Volume<Float, Spectrum>::local_majorants(size_t /*resolution_factor*/, ScalarFloat /*value_scale*/) {
    NotImplementedError("local_majorants");
}

MTS_VARIANT typename Volume<Float, Spectrum>::ScalarVector3f
Volume<Float, Spectrum>::voxel_size() const {
    // Extract the scale from the to_world matrix, assuming an affine transformation.
    ScalarTransform4f to_world = m_to_local.inverse();
    ScalarVector3f scale(
        ek::norm(to_world.matrix.x()),
        ek::norm(to_world.matrix.y()),
        ek::norm(to_world.matrix.z())
    );
    return ek::rcp(ScalarVector3f(resolution())) * scale;
}

MTS_VARIANT typename Volume<Float, Spectrum>::ScalarVector3i
Volume<Float, Spectrum>::resolution() const {
    return ScalarVector3i(1, 1, 1);
}

//! @}
// =======================================================================

MTS_IMPLEMENT_CLASS_VARIANT(Volume, Object, "volume")

MTS_INSTANTIATE_CLASS(Volume)
NAMESPACE_END(mitsuba)
