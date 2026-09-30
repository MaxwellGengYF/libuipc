#pragma once
/********************************************************************
 * @file contact_coeff.h
 * @brief Per-pair contact coefficient (port of
 * src/backends/cuda/contact_system/contact_coeff.h, 14 lines).
 *
 * Pure data: the normal stiffness `kappa` and the friction coefficient `mu`
 * of one *ordered* contact-model pair.  It is a `LUISA_STRUCT`-registered
 * device element (a flat `lud::LDeviceBuffer<ContactCoeff>` is the port's
 * spelling of the reference's `DeviceBuffer2D<ContactCoeff>` — see
 * contact_system/global_contact_manager.h).  Nothing else in the reference
 * file, so the transcription is the two fields and the constructor default.
 *********************************************************************/
#include <type_define.h>

namespace uipc::backend::luisa
{
class ContactCoeff
{
  public:
    // normal stiffness
    Float kappa = 0.0;
    // friction coefficient
    Float mu = 0.0;
};
}  // namespace uipc::backend::luisa

// LuisaCompute struct registration (global scope, before first use): a flat
// `lud::LDeviceBuffer<ContactCoeff>` is a kernel argument, so the element type must
// be a reflected LC struct (cookbook §3.2 rule B0 / type_define.h's order rules).
LUISA_STRUCT(uipc::backend::luisa::ContactCoeff, kappa, mu){};
