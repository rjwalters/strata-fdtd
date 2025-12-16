---
title: "Strata: Layered Metamaterial Loudspeaker System"
subtitle: "Engineering Specification"
---

# Objective

Design a flagship audiophile loudspeaker system where the **enclosure itself is an engineered acoustic impedance network**—not a passive box, but an active participant in sound reproduction.

::: {#strata-hero .image}
**Layered metamaterial loudspeaker with exposed acoustic labyrinth**

Dramatic three-quarter view of a floor-standing tower speaker with a striking industrial aesthetic. The cabinet is partially cut away to reveal complex internal acoustic chambers—a labyrinth of precisely machined cavities and serpentine channels carved into laminated MDF layers. The exposed cross-section shows visible strata (wood grain layers) with geometric patterns of Helmholtz resonators at varying scales.

A ribbon tweeter sits at ear height, with a midrange driver below it surrounded by an intricate metachamber. The lower section houses a large woofer. Where the cabinet opens to reveal internals, the geometry is beautiful and deliberate—not random foam stuffing, but engineered precision.

Lighting emphasizes the depth and complexity of the internal structure. Modern audiophile aesthetic with matte black exterior contrasting against warm wood tones of exposed laminations. No text labels.

- **style**: photo-realistic
- **aspect**: 2:1
:::

**NOTE:** This specification describes a design concept for a high-end loudspeaker system. The acoustic metamaterial approach draws on established physics (Helmholtz resonance, transmission line theory) but the specific implementation requires simulation validation and prototype measurement before performance claims can be verified.

# Executive Summary

Strata is a **layered metamaterial loudspeaker system** that replaces conventional enclosure damping (stuffing, foam) with **geometry-defined acoustic impedance networks**. The enclosure's internal structure is engineered at subwavelength scales to absorb, redirect, and terminate rear-wave energy before it can color the reproduced sound.

**Core innovation:**

- **Acoustic metamaterials by construction**: Helmholtz cavities, serpentine channels, and tapered terminations are CNC-machined directly into laminated MDF slices—no separate absorber materials required

- **Manufacturing-first design**: Digital fabrication (laser/CNC cutting of 6 mm MDF sheets, stacked and bonded) enables complex internal geometries at moderate cost

- **Exposed engineering aesthetic**: Cutaway regions reveal the internal acoustic labyrinth, making the technology visible and distinctive

**System configuration:**

- **Tower speakers** (pair): 3-way active (woofer / midrange / ribbon tweeter), each driver with purpose-designed acoustic cavity
- **Mono bass module** (optional): Dedicated subwoofer cabinet extending response below tower capability

**Target market:** High-end audiophile systems ($8,000–15,000 retail for tower pair). The exposed metamaterial aesthetic and engineering-forward design language differentiate from both conventional box speakers and exotic enclosure materials.

**Key hypothesis:** By engineering the enclosure's acoustic impedance rather than relying on bulk absorbers, we can achieve lower stored energy, faster decay times, and reduced coloration—particularly in the critical midrange band where enclosure effects are most audible.

**Target performance (vs. conventional stuffed enclosure of equivalent volume):**

<div class="center">

| **Metric** | **Conventional (reference)** | **Strata (target)** | **Measurement Method** |
|:-----------|:-----------------------------|:--------------------|:-----------------------|
| Decay to –20 dB | ~25 ms typical | < 10 ms | Impulse response (Schroeder integral) |
| Stored energy ratio | 1.0 | < 0.3 | $E(t) = \int_V p^2 dV$ integration |
| Reflection coefficient | ~0.25 | < 0.1 | Near-field driver measurement |
| Midrange coloration (300–1.2k) | Baseline | –6 dB peak reduction | Transfer function $H(f)$ analysis |

</div>

These targets are derived from acoustic theory and published metamaterial research; validation requires simulation and prototype measurement. The specification explicitly defines what measurements would falsify the core hypothesis: if stored energy reduction is less than 50% or decay time improvement is less than 2×, the metamaterial approach does not justify its complexity over conventional damping.

# Design Philosophy & Goals

## Geometry as the Primary Acoustic Material

Traditional loudspeaker design treats the enclosure as a nuisance to be damped. Strata treats the enclosure as an **acoustic component whose impedance is intentionally designed**.

This represents a fundamental shift in approach:

- **Conventional**: Reduce reflections with stuffing (fiberglass, foam, wool)
- **Strata**: Shape impedance continuously with geometry

This reframing enables:

- **Predictable behavior**: Geometry is deterministic; stuffing varies batch-to-batch
- **Simulation-driven design**: CAD geometry maps directly to acoustic simulation
- **Longevity**: No aging, settling, or degradation of damping materials
- **Aesthetic exposure**: Function becomes form—the acoustic mechanism is visible

## Approximate Free-Space Loading in a Finite Volume

The ideal loudspeaker driver would radiate into free space—infinite baffle, no enclosure reflections, no stored energy. Practical loudspeakers compromise this ideal: the enclosure interacts with the driver's rear radiation, creating resonances, delayed reflections, and coloration.

The core acoustic goal is to approximate **reflectionless rear loading** for each driver within a compact enclosure. Rather than attempting to:

- Eliminate reflections entirely (impossible in finite volume)
- Apply heavy loss everywhere (inefficient, causes temporal smearing)

Strata aims to:

- **Gradually transform impedance** from the driver cone to a resistive termination
- **Distribute loss spatially** so energy is dissipated along the path, not stored

This concept borrows from horn theory and muffler design, but applied *inward* and at subwavelength scales—matching the driver to an absorbing termination rather than to open air.

## Geometry Over Stuffing

Conventional loudspeaker enclosures use bulk absorbers to damp internal resonances. This approach has inherent limitations:

<div class="center">

| **Approach** | **Mechanism** | **Limitations** |
|:-------------|:--------------|:----------------|
| Fiberglass/wool | Viscous friction | Effective only above ~200 Hz; adds mass |
| Foam | Viscous/thermal | Inconsistent properties; degradation over time |
| Sand/shot filling | Mass loading | Addresses panel vibration, not air modes |
| Constrained-layer damping | Shear dissipation | Panel modes only; no effect on cavity resonance |

</div>

Acoustic metamaterials offer an alternative: **geometry-defined absorption** that works by designing cavity/neck dimensions to create resonant absorption at target frequencies. Arrays of detuned Helmholtz resonators can provide broadband absorption without bulk materials. Recent research demonstrates that hybrid resonance between non-uniformly partitioned Helmholtz resonators can achieve near-perfect sound absorption at target frequencies by breaking geometrical symmetry between adjacent resonators (Choi & Jeon, 2024).

## Active System + DSP

Strata is designed as an **active loudspeaker system** with dedicated amplification and DSP for each driver. This enables:

- **Optimized crossovers**: FIR filters with linear phase and steep slopes
- **Driver linearization**: Correction for driver-specific response anomalies
- **Room correction**: Integration with measurement-based EQ
- **Thermal protection**: Real-time power management
- **Bass alignment**: Phase-coherent integration with optional bass module

The active topology also eliminates passive crossover components (inductors, capacitors) that would occupy valuable internal volume needed for acoustic metamaterial structures. This is a critical architectural constraint: the metamaterial approach may be incompatible with passive crossover designs due to volume competition.

## Visual Transparency of Engineering

Where most loudspeakers hide their internals, Strata exposes them. Strategic cutaway regions reveal the acoustic labyrinth, making the engineering visible:

- **Functional honesty**: The visible geometry is not decorative—it is the working acoustic system
- **Differentiation**: Instantly distinguishable from conventional box speakers
- **Conversation piece**: The complexity invites inspection and explanation
- **Quality signal**: Visible precision implies overall build quality

This aesthetic requires that exposed structures be finished to display standards—the engineering must look as good as it works.

# System Architecture Overview

## Tower Speaker Architecture

Each tower is a **3-way active loudspeaker** with separate acoustic cavities for each driver:

::: {#strata-tower-architecture .figure}
**Tower speaker driver and cavity layout**

Vertical cross-section of tower speaker showing three distinct acoustic zones:

**Tweeter Section** (top, ~200mm height):
- Ribbon tweeter with shallow sealed back chamber
- Minimal cavity complexity—ribbons are dipole or near-dipole

**Midrange Section** (middle, ~400mm height):
- 5–6" midrange driver
- Elaborate metachamber with Helmholtz network
- Serpentine channels with progressive taper
- This is the primary metamaterial application zone

**Woofer Section** (bottom, ~600mm height):
- 8–10" woofer
- Larger sealed or vented cavity
- Metamaterial treatment for upper bass/lower midrange
- Optional metaport for vented alignment

Cutaway view shows internal structure in midrange and woofer sections. Labels indicate driver positions, cavity boundaries, and approximate volume allocations.
:::

### Crossover Topology

<div class="center">

| **Driver** | **Passband** | **Crossover** | **Slope** |
|:-----------|:-------------|:--------------|:----------|
| Ribbon tweeter | 3 kHz – 20 kHz+ | HP @ 3 kHz | 48 dB/oct (FIR) |
| Midrange | 300 Hz – 3 kHz | BP 300/3k | 48 dB/oct (FIR) |
| Woofer | 40 Hz – 300 Hz | LP @ 300 Hz | 48 dB/oct (FIR) |

</div>

The steep FIR slopes minimize overlap between drivers, reducing interference effects and allowing each driver's cavity to be optimized for its operating band.

### Role of the Tower

The tower handles the full frequency range for typical listening. It is designed for:

- **Moderate listening levels**: 85–95 dB SPL at listening position
- **Room sizes**: Small to medium rooms (15–40 m²)
- **Bass extension**: –3 dB at ~40 Hz (room-dependent)

For larger rooms or higher SPL requirements, the optional bass module extends capability.

## Optional Mono Bass Module

A dedicated mono subwoofer cabinet provides:

- **Extended bass**: –3 dB at 25 Hz or below
- **Increased SPL capability**: 10–15 dB headroom over towers alone
- **Optimal placement**: Bass module positioned for room mode coupling, independent of tower placement for imaging

### Why Mono?

Below ~80 Hz, human localization cues are minimal. A single, well-placed bass module provides:

- **Simplified room integration**: One bass source to optimize
- **Cost efficiency**: One large cabinet vs. two
- **Flexible positioning**: Placed for bass response, not stereo image
- **Power consolidation**: Single high-power amplifier channel

The bass module receives a mono sum of the stereo signal, low-passed and phase-aligned to the towers.

# Manufacturing Concept: Laminated MDF Construction

## Lamination as an Enabler, Not a Compromise

The slice-based MDF approach is not merely a cost or tooling workaround—it is what **makes the acoustic concept possible**.

Lamination enables:

- **Arbitrary 3D internal topology**: Enclosed labyrinths, branching resonator trees
- **Deep subwavelength features**: 6 mm resolution for mid/high frequency metamaterials
- **Distributed networks**: Side-branch resonators along continuous paths
- **High internal surface area**: For controlled viscous and thermal losses

Traditional CNC machining struggles to produce:

- Enclosed chambers accessible only through small necks
- Smooth impedance tapers with internal side-branches
- Complex branching geometries in a single setup

Laser-cut lamination makes these routine—each slice is a 2D problem, and the 3D structure emerges from stacking.

## Material Choice: 6 mm MDF

Medium-density fiberboard (MDF) is the foundation material:

- **Homogeneous**: Consistent density and properties in all directions
- **Machinable**: Clean cuts with laser, router, or waterjet
- **Acoustically dead**: High internal damping, low resonance Q
- **Economical**: Commodity material, widely available
- **Bondable**: Standard wood glues create strong laminations

The 6 mm thickness is chosen for:

- **Resolution**: Fine enough to create meaningful acoustic features
- **Structural**: Thick enough for robust laminations
- **Standard**: Common sheet good thickness

## Slice-Based Lamination Process

The enclosure is built from stacked horizontal slices:

::: {#strata-lamination-process .figure}
**Slice-based lamination construction sequence**

Four-panel sequence showing construction process:

**Panel 1 - Digital Design**:
CAD model of enclosure with internal cavities and channels.
Each horizontal slice extracted as 2D cutting path.

**Panel 2 - Sheet Cutting**:
6mm MDF sheets on laser/CNC bed.
Multiple slices nested for efficient material use.
Cut lines show internal cavity profiles.

**Panel 3 - Stacking & Bonding**:
Exploded view of slices being stacked.
Registration pins/dowels align layers.
Glue application between layers.
Clamping pressure during cure.

**Panel 4 - Finished Structure**:
Completed enclosure cross-section.
Internal channels visible.
External surfaces ready for finishing.

Arrows show progression through stages.
:::

### Digital Fabrication Advantages

- **Complexity is free**: Intricate internal geometry adds no fabrication cost
- **Repeatability**: CNC ensures identical units
- **Iteration**: Design changes require only new cut files
- **Material efficiency**: Nesting software optimizes sheet utilization
- **Documentation**: Cut files serve as manufacturing specification

### Registration and Assembly

Precise layer alignment is critical:

- **Dowel pins**: Each layer includes alignment holes for assembly jigs
- **Laser indexing**: Etched marks for visual alignment verification
- **Clamping fixtures**: Custom jigs ensure even pressure during bonding
- **Glue selection**: PVA or polyurethane for appropriate working time

### Constrained-Layer Damping as Structural Side-Effect

The laminated structure inherently provides constrained-layer damping—a beneficial side-effect of the manufacturing approach:

Each glue interface between slices acts as a constrained layer:

- **Shear interface**: Glue line between layers dissipates vibrational energy through shear deformation
- **Mass loading**: Accumulated layers create high surface mass
- **Distributed stiffness**: No single resonant panel mode dominates
- **Upward mode shift**: Panel resonances are shifted to higher frequencies where they're less audible

This reduces the need for:

- Heavy internal bracing (which consumes acoustic volume)
- Lossy external panel treatments
- Exotic cabinet materials

The result is that structural stiffness and damping increase together—the same lamination that enables acoustic complexity also improves mechanical behavior.

**Quantification required**: The constrained-layer damping benefit is asserted but unquantified. Panel mode specification (target first mode > 500 Hz) and measured damping factor (target Q < 5 for primary modes) should be validated through modal analysis of prototype structures.

Additional damping can be added by alternating MDF with viscoelastic interlayers at strategic positions for extreme applications.

# Acoustic Metamaterial Framework

## Helmholtz Networks as Effective Media

A Helmholtz resonator is a cavity connected to the main volume through a narrow neck. At its resonant frequency, it presents high acoustic impedance to the main volume—effectively absorbing energy at that frequency.

Rather than tuning a single Helmholtz resonator to a problem frequency, Strata uses **distributed, detuned networks**:

- Many cavities with slightly different volumes
- Many necks with slightly different inertance
- No single sharp resonance—smooth broadband impedance behavior

At wavelengths much larger than feature size, this behaves as an **effective medium** with tunable properties:

- Effective compliance (determined by total cavity volume)
- Effective resistance (determined by neck geometry and viscous losses)
- Frequency-dependent impedance (shaped by the resonator distribution)

This is the defining metamaterial concept in the system—many small structures combine to create bulk acoustic properties that don't exist in the constituent materials.

::: {#strata-helmholtz-network .figure}
**Helmholtz resonator array as broadband absorber**

Left side: Single Helmholtz resonator schematic showing:
- Cavity volume V
- Neck length L and area S
- Resonant frequency formula: f₀ = (c/2π)√(S/VL)
- Impedance curve showing absorption peak

Right side: Array of detuned resonators:
- Multiple cavities with different V, L, S parameters
- Resonant frequencies distributed across target band
- Combined absorption curve showing broadband effect
- Frequency axis labeled 200 Hz – 2 kHz

Key insight: By tuning many small resonators across the frequency band, we create an effective absorbing medium without bulk materials.
:::

### Subwavelength Design

For metamaterial behavior, cavity dimensions must be small relative to wavelength:

<div class="center">

| **Frequency** | **Wavelength** | **Target Cavity Scale** |
|:--------------|:---------------|:------------------------|
| 200 Hz | 1.7 m | < 170 mm |
| 500 Hz | 680 mm | < 70 mm |
| 1 kHz | 340 mm | < 35 mm |
| 2 kHz | 170 mm | < 17 mm |

</div>

The 6 mm MDF lamination naturally creates features in the 6–60 mm range, well-suited for metamaterial structures targeting 500 Hz – 3 kHz—precisely the midrange band where enclosure coloration is most audible.

### Acoustic Parameters from Geometry

Each Helmholtz cell contributes three acoustic parameters:

- **Compliance** ($C_a$): Proportional to cavity volume
- **Inertance** ($M_a$): Proportional to neck length, inversely to neck area
- **Resistance** ($R_a$): From viscous losses in the neck

By arranging cells in series and parallel, we create acoustic networks with designed impedance characteristics—analogous to electrical filter design.

### Deep Subwavelength Design via Kerf-Limited Geometry

Laser kerf and material thickness define a **minimum feature scale** that is still deeply subwavelength across most of the audible band.

<div class="center">

| **Feature** | **Typical Dimension** | **Subwavelength Below** |
|:------------|:---------------------|:-----------------------|
| Laser kerf | 0.2–0.5 mm | ~70 kHz |
| Minimum slot width | 2–3 mm | ~10 kHz |
| Lamination thickness | 6 mm | ~6 kHz |
| Minimum practical cavity | 12–18 mm | ~2–3 kHz |

</div>

This enables:

- **Viscous and thermal boundary losses** to dominate at high frequencies (intentional damping without fibrous materials)
- **Controlled damping** through geometry rather than bulk absorber properties
- **Repeatable, simulation-compatible structures** (no material variability)

Critically, the smallest features are *not* used for main airflow—they are used intentionally as **loss elements** in side branches where high velocity through narrow gaps dissipates energy.

## Tapered Rear-Wave Impedance Strategy

The driver's rear radiation sees the enclosure as an acoustic load. Our goal is to create a **progressive impedance transition** from the cone to a final termination:

::: {#strata-impedance-taper .figure}
**Tapered impedance matching from driver to termination**

Horizontal schematic showing rear-wave path:

**Driver Cone** (left):
- High velocity, low impedance
- Radiation into enclosure

**Transition Zone**:
- Gradually narrowing channels
- Progressive impedance increase
- Helmholtz cells branch off main path
- Each cell absorbs energy at its resonant frequency

**Termination** (right):
- High impedance, low velocity
- Remaining energy absorbed by final cells
- Minimal reflection back toward driver

Below: Impedance magnitude plot showing smooth transition from Z_cone to Z_termination. Reflection coefficient decreases along path.

**Key principle**: Horn/transmission-line concepts applied inward—matching the cone to an absorbing termination rather than to open air.
:::

### Impedance Transformation, Not Reflection Trapping

The enclosure is **not** designed to "trap" sound or absorb it through brute force. Instead, the geometry creates a controlled impedance transformation:

- **Near the cone**: Lower impedance matches the driver's radiation
- **Along the path**: Gradually increasing impedance via narrowing channels
- **Side branches**: Helmholtz cells absorb energy at their resonant frequencies
- **At termination**: Remaining energy dissipated resistively into final absorbing cells

This reduces:

- Pressure reflection back onto the diaphragm (which causes coloration)
- Modal buildup (standing waves between driver and walls)
- Time-domain ringing (stored energy releasing after the input stops)

### Horn Theory, Inverted

The mathematics of this approach is borrowed from horn design, but with inverted goals:

<div class="center">

| **Aspect** | **Traditional Horn** | **Strata Impedance Taper** |
|:-----------|:--------------------|:--------------------------|
| Direction | Driver → free space | Driver → termination |
| Goal | Maximize radiation | Minimize reflection |
| Termination | Open air (low Z) | Resistive absorber (matched Z) |
| Efficiency | Increase output | Increase absorption |

</div>

Traditional horns match a driver to free space to increase radiation efficiency. Strata matches a driver to a resistive termination to *reduce* reflection efficiency. The wave equations are analogous, but the boundary conditions—and therefore the design intent—are inverted.

### Goals of the Impedance Strategy

1. **Low reflection**: Energy traveling away from cone should not bounce back
2. **Distributed absorption**: Energy removed throughout the path, not just at termination
3. **Broadband operation**: Effective across the driver's full passband
4. **Low stored energy**: Minimal cavity resonance to color the sound

This is essentially a **transmission line** philosophy, but instead of an open or closed termination, we engineer an absorbing termination.

# Driver-Specific Acoustic Cavities

## Decoupling Design Constraints

By giving each driver its own isolated enclosure volume, Strata decouples design constraints that are normally in tension:

- **The midrange is no longer compromised by bass volume needs**: Midrange cavity geometry is optimized purely for coloration suppression, not shared with woofer alignment requirements
- **The woofer chamber is not polluted by midrange reflections**: No energy coupling between drivers through shared internal volume
- **The tweeter is mechanically isolated**: Vibration from lower drivers doesn't reach the ribbon

Each cavity becomes a **single-purpose acoustic device**, optimized for its operating band without compromise.

This isolation is enforced structurally: the laminated MDF construction creates solid barriers between driver sections, with no shared air paths.

## Woofer (Midbass) Chamber

The woofer operates in a sealed or vented alignment, with metamaterial treatment focused on the upper bass / lower midrange transition region (150–400 Hz).

### Volume and Alignment

<div class="center">

| **Parameter** | **Sealed Alignment** | **Vented Alignment** |
|:--------------|:--------------------|:---------------------|
| Net volume | 35–50 L | 50–70 L |
| System Q (Qtc) | 0.65–0.75 | N/A |
| Tuning frequency (Fb) | N/A | 32–40 Hz |
| –3 dB point (typical) | 45–55 Hz | 35–42 Hz |

</div>

The sealed alignment offers simpler construction and better transient response. The vented alignment extends bass response at the cost of increased complexity and port noise considerations.

### Metamaterial Application

For the woofer cavity, metamaterial structures target:

- **150–400 Hz**: Upper bass modes that could interact with midrange
- **Standing wave suppression**: Break up length/width/height modes
- **Port channel treatment**: If vented, the port path includes absorption cells

The scale of required structures (50–100 mm features) is achievable with the 6 mm lamination approach, though fewer layers per acoustic feature limits resolution compared to the midrange application.

## Midrange "Metachamber": The Highest-ROI Target

The midrange cavity is the **highest-value application** of metamaterial technology—where the engineering investment pays off most directly in audible improvement.

Human hearing is most sensitive to coloration in the midrange (300 Hz – 3 kHz):

- **Perceptual sensitivity**: The ear is exquisitely tuned to detect artifacts in this band (voice, instruments)
- **"Box sound" lives here**: The characteristic coloration of conventional enclosures is concentrated in the midrange
- **Wavelength/feature match**: 300 Hz – 1.2 kHz wavelengths (280–1130 mm) match well to achievable cavity dimensions (20–100 mm at λ/10)
- **Conventional damping fails**: Fiberglass and foam are least effective in the lower midrange

This is where:

- Improvements are **immediately audible** to any listener
- Simulation effort produces **measurable results**
- The metamaterial approach has **maximum advantage** over stuffing

The midrange metachamber therefore receives:

- The most complex internal geometry
- The densest Helmholtz resonator network
- The most careful optimization across bandwidth

### Target Frequency Band

The midrange driver operates 300 Hz – 3 kHz, but the metamaterial treatment focuses on the **lower half** (300 Hz – 1.2 kHz) where:

- Wavelengths are long enough for cavity effects
- Conventional damping is least effective
- Vocal fundamentals and presence region live

Above 1.2 kHz, wavelengths become shorter than practical cavity dimensions, and conventional absorbers become effective.

### Metachamber Architecture

::: {#strata-metachamber .figure}
**Midrange metachamber with Helmholtz network**

Cross-section of midrange cavity showing:

**Primary Channel**:
Serpentine path from driver rear to termination.
Channel width tapers from ~50mm at driver to ~15mm at termination.
Total path length ~400mm (folded into available height).

**Helmholtz Branches**:
Multiple cavity/neck pairs branch off main channel.
Larger cavities (lower frequency) near driver end.
Smaller cavities (higher frequency) near termination.
Approximately 8–12 cells distributed along path.

**Cavity Dimensions Table**:
| Position | Target f₀ | Cavity | Neck |
|----------|-----------|--------|------|
| Near driver | 350 Hz | 40×40×30mm | 6×10mm |
| Mid-path | 600 Hz | 25×25×25mm | 6×8mm |
| Near term | 1000 Hz | 15×15×20mm | 4×6mm |

**Final Termination**:
Absorptive wedge geometry with distributed small cells.
Remaining energy dissipated as heat.

Arrows show acoustic energy flow from driver through network.
:::

### Suppression of Cavity Coloration

The metachamber design targets specific failure modes of conventional midrange enclosures:

1. **Standing waves**: Broken up by serpentine path and irregular geometry
2. **First mode resonance**: Helmholtz cells tuned to absorb
3. **Reflections to cone**: Tapered impedance minimizes bounce-back
4. **Delayed energy**: Multiple absorption points prevent energy storage

The goal is a midrange cavity that behaves, from the driver's perspective, like an infinite baffle—no reflections, no resonances, no stored energy.

## Tweeter Back Chamber

Ribbon tweeters are typically dipole or near-dipole radiators with open back structures. The "chamber" is minimal:

- **Isolation**: Acoustic separation from midrange cavity
- **No coloration concerns**: Operating frequencies (>3 kHz) have wavelengths too short for cavity effects
- **Mechanical mounting**: Secure driver attachment and wiring access

If a dome tweeter is used instead, a small sealed chamber (50–100 mL) with light damping is sufficient.

# Exposed Metaport & Cutaway Industrial Design

## Functionally Honest Aesthetics

The exposed cutaway is not decorative abstraction. It is:

- **A literal cross-section** of the working acoustic network
- **Visual communication of function**: The complexity is not ornament—it *is* the technology
- **Instant differentiation**: No other loudspeaker looks like this because no other loudspeaker works like this

This is rare in audio, where internals are usually hidden or cosmetically disguised. Strata makes the engineering visible because the engineering *is* the product.

## The Metaport as a 3D Acoustic Radiator

Traditional ports are essentially 1D devices: a tube connecting the enclosure to free space.

The Strata metaport is a **3D acoustic radiator**:

- Fed by an internal trunk (serpentine path through laminations)
- Split into many low-velocity outlets (distributed apertures)
- Shaped to reduce turbulence and tonal noise
- Visible as a sculptural element on the cabinet exterior

The grille is no longer a perforated plate—it is a **radiating geometry** where the outlet structure itself is acoustically designed.

If the woofer uses a vented alignment, the port path becomes an opportunity for exposed metamaterial design:

::: {#strata-metaport .figure}
**Metaport outlet with diffuser geometry**

Three views of the metaport design:

**Internal Path** (cutaway view):
Port channel begins at woofer cavity.
Serpentine routing through laminated layers.
Helmholtz cells along path absorb turbulence frequencies.
Progressive flare approaching outlet.

**Outlet Geometry** (detail view):
3D-printed or CNC-machined diffuser insert.
Multiple small openings vs single large port.
Smooth radius edges minimize turbulence.
Integrated into cabinet side or rear.

**External Appearance** (beauty shot):
Metaport visible on cabinet exterior.
Exposed wood strata around opening.
Diffuser in contrasting material (metal or dark polymer).
Functional component as design element.

Key insight: Port noise comes from turbulence at high air velocity. Distributed outlets reduce velocity; internal absorption damps turbulence that does occur.
:::

### Port Noise Analysis

Port noise is a critical validation requirement. The distributed metaport design must maintain air velocity below turbulence thresholds:

**Velocity calculation for 110 dB SPL at 40 Hz** (bass module specification):

At 110 dB SPL, the acoustic volume velocity through a bass reflex port is substantial. For turbulence-free operation:

<div class="center">

| **Parameter** | **Requirement** | **Derivation** |
|:--------------|:----------------|:---------------|
| Target velocity | < 5 m/s | Turbulence onset threshold |
| Margin velocity | < 10 m/s | Acceptable with some noise |
| Required port area | ~80–120 cm² | At rated output |
| Distributed outlet count | 8–16 | Each 5–10 cm² |

</div>

**Validation protocol**:
1. Sine sweep excitation at rated SPL
2. Microphone positioned at port outlet (near-field)
3. Pass criterion: Noise floor > 20 dB below fundamental at all frequencies
4. Test at 1.5× rated SPL to establish headroom

If port noise exceeds thresholds, mitigation options include:
- Increased total port area (more distributed outlets)
- Longer serpentine path (more absorption before outlet)
- Sealed alignment (eliminates port entirely)

## Functional vs. Sealed Cutaway Regions

The cabinet includes several categories of exposed geometry:

<div class="center">

| **Region** | **Acoustic Function** | **Visual Treatment** |
|:-----------|:---------------------|:---------------------|
| Metaport outlet | Active port termination | Open, finished edges |
| Midrange cutaway | Display of metachamber | Sealed with clear window |
| Woofer cutaway | Display of internal structure | Sealed with clear window |
| Decorative reveals | None (visual only) | Exposed strata, finished |

</div>

**Sealed cutaway windows** allow viewing of internal structure without compromising acoustic isolation. These can be:

- Acrylic panels set into routed recesses
- Frameless bonded glass for seamless appearance
- Removable for demonstration/photography

## Separation of Flow and Loss

A critical non-obvious design rule: **main airflow paths are smooth and generous; lossy microfeatures live in side branches.**

<div class="center">

| **Path Type** | **Geometry** | **Function** |
|:--------------|:-------------|:-------------|
| Main flow channel | Wide, smooth, generous radii | Bulk air movement, low resistance |
| Side-branch resonators | Narrow necks, small cavities | Energy absorption at target frequencies |
| Termination cells | Fine slots, high surface area | Final energy dissipation |

</div>

This separation prevents:

- **Resistive choking**: Main path remains low-impedance for driver loading
- **Compression artifacts**: No dynamic restriction at high SPL
- **Noise at high SPL**: Turbulence occurs in lossy branches, not main flow

The visible complexity does not imply acoustic restriction—the elaborate geometry is for *absorption*, not *obstruction*.

## Acoustic Constraints for Quiet Operation

Exposed port structures must not introduce noise:

- **Air velocity limit**: Peak velocity < 5–10 m/s to avoid turbulence
- **Distributed outlets**: Multiple small openings reduce velocity for given flow
- **Smooth radii**: No sharp edges in airflow path
- **Helmholtz filtering**: Absorption cells along path damp turbulence before outlet

Port noise testing is a critical validation step—any audible contribution defeats the design intent.

# Surface Treatment & Finish Strategy

## MDF Edge Hardening and Sealing

Raw MDF edges absorb moisture and show a fuzzy texture. Exposed lamination edges require treatment:

- **Hardening**: Thin cyanoacrylate (CA) wicking hardens fibers
- **Sealing**: Multiple thin shellac or lacquer coats prevent absorption
- **Sanding**: Progressive grits (220 → 400 → 600) between coats
- **Final finish**: Matte or satin clear coat for durability

The goal is visible wood strata with sealed, durable surfaces.

## Surface Class Differentiation

The cabinet has three surface categories requiring different finishing approaches:

### Smooth Body Surfaces (Class A)

- External cabinet faces
- Conventional wood finishing: fill, sand, prime, topcoat
- Options: high-gloss lacquer, satin black, wood veneer
- Standard furniture finishing techniques apply

### Exposed Cutaway Regions (Class B)

- Visible lamination strata
- Show material character while ensuring durability
- Edge hardening + clear finish
- Accept minor tooling marks as part of aesthetic

### Functional Vent Areas (Class C)

- Port outlets, ventilation paths
- Must not add acoustic artifacts
- Smooth, non-resonant materials
- May use metal or polymer inserts

## Finish Options

<div class="center">

| **Finish** | **Appearance** | **Complexity** | **Cost** |
|:-----------|:---------------|:---------------|:---------|
| Matte black + natural strata | High contrast, industrial | Medium | $$ |
| High-gloss lacquer | Premium conventional | High | $$$ |
| Natural MDF + clear | Raw/honest aesthetic | Low | $ |
| Wood veneer + strata accents | Traditional + modern | High | $$$ |

</div>

The recommended approach is **matte black exterior with natural finished strata** in cutaway regions—maximum contrast between smooth cabinet and complex internal geometry.

# Simulation & Optimization Approach

## Geometry-First Simulation Philosophy

The system is simulated directly from CAD geometry:

- **Air vs. rigid solid voxels**: No material property guessing or absorber characterization
- **Time-domain wave propagation**: Full acoustic physics, not modal approximations
- **Broadband excitation**: Impulse response captures all frequencies simultaneously

This approach avoids the pitfalls of traditional enclosure modeling:

- No frequency-by-frequency tuning that misses interactions
- No modal cherry-picking that ignores broadband behavior
- No overfitting to narrow test conditions

The geometry *is* the model—CAD files translate directly to simulation input.

**Known limitation**: FDTD models may not fully capture viscous and thermal losses in narrow channels. The simulation framework must include validated loss models for neck geometries, and simulation-to-measurement correlation is required before relying on simulation alone. See Phase 2 validation milestone.

## Numerical Modeling Strategy

### Voxelized FDTD Acoustics

Finite-difference time-domain (FDTD) simulation models acoustic wave propagation through the enclosure geometry:

- **Domain**: 3D voxel grid representing air and solid regions
- **Resolution**: Voxel size determines maximum simulated frequency
- **Excitation**: Driver modeled as velocity source
- **Output**: Pressure field evolution, frequency response, impulse response

### Resolution Considerations

<div class="center">

| **Frequency Limit** | **Voxel Size** | **Cells for 500mm** | **Memory (float)** |
|:--------------------|:---------------|:--------------------|:-------------------|
| 1 kHz | 17 mm | 30 | ~100 KB |
| 2 kHz | 8.5 mm | 60 | ~1 MB |
| 4 kHz | 4.2 mm | 120 | ~7 MB |
| 8 kHz | 2.1 mm | 240 | ~55 MB |

</div>

Full 3D simulation of the midrange cavity at 4 kHz resolution is computationally tractable on modern workstations. Higher frequencies (tweeter) can use simplified 2D or analytical models.

### Separate Resolution Regimes

- **Woofer cavity**: Coarse resolution (2 kHz limit), large domain
- **Midrange cavity**: Medium resolution (4 kHz limit), focus on metamaterial detail
- **Tweeter**: Analytical or 2D model, minimal simulation need

## Objective Metrics

Rather than optimizing frequency response directly (which can lead to narrow-band fixes that miss the broader problem), Strata focuses on **root-cause metrics** that correlate with perceived quality.

### Stored Energy as the Primary "Badness" Metric

The central optimization target is **stored acoustic energy** in the enclosure:

$$E(t) = \int_V p^2(x,t) dV$$

Stored energy correlates strongly with:

- **Perceived "boxiness"**: Energy releasing after the input stops
- **Transient smear**: Impulse response tails
- **Midrange coloration**: Resonant peaks and delayed reflections

Target: –20 dB decay in < 10 ms (faster than room decay, ensuring enclosure artifacts are masked by room acoustics)

A low stored-energy design is inherently a good design—the frequency response follows.

### Rear Pressure Reflection

Energy returning to driver cone from enclosure:

$$R = \frac{\int |p_{reflected}|^2 dt}{\int |p_{incident}|^2 dt}$$

Target: $R < 0.1$ (< 10% energy reflected)

This directly measures how well the impedance taper is working. High reflection means energy is bouncing back to the cone, causing coloration.

### Resonance Peak Control

Transfer function from driver to pressure at critical points:

$$H(f) = \frac{P_{internal}(f)}{P_{driver}(f)}$$

Target: No peaks > +6 dB relative to passband average

### Robustness to Tolerances

Sensitivity of performance metrics to dimensional variations:

$$\frac{\partial R}{\partial \Delta} \cdot \sigma_\Delta$$

Target: < 10% performance variation for ±1 mm tolerance

## Optimization Methodology

### Parametric Geometry Families

Rather than optimizing arbitrary geometry, define parametric families:

- **Serpentine channel**: Width taper, fold count, total length
- **Helmholtz array**: Count, frequency spacing, Q values
- **Port geometry**: Cross-section, flare rate, diffuser pattern

Optimization explores parameter space rather than arbitrary shapes.

### ML-Style Multi-Objective Optimization

The geometry is treated like a hyperparameter space:

- **Parameters**: Taper rates, resonator distributions, loss densities, channel widths
- **Constraints**: Manufacturability limits, volume budgets, minimum feature sizes
- **Objectives**: Stored energy, reflection coefficient, decay time, robustness

Optimization searches for **Pareto-optimal families**, not a single "perfect" shape:

- **Nelder-Mead simplex**: Fast local optimization from promising starts
- **Pareto frontier mapping**: Trade-off curves between competing objectives
- **Surrogate models**: Fast approximations (neural networks or Gaussian processes) for coarse search

This mirrors modern ML system design:

- Treat the geometry as a search space, not a single design
- Optimize for robustness, not just peak performance
- Accept multiple good solutions rather than one brittle optimum

The result is a family of designs that all meet requirements, allowing selection based on manufacturing or aesthetic preferences.

### Coarse-to-Fine Pipeline

1. **Coarse grid search**: Low-resolution simulation, many candidates
2. **Promising region refinement**: Higher resolution on top candidates
3. **Final validation**: Full-resolution simulation of selected designs
4. **Sensitivity analysis**: Check robustness to variations

# Driver Selection Strategy

## Performance Requirements by Band

<div class="center">

| **Driver** | **Passband** | **Key Requirements** |
|:-----------|:-------------|:---------------------|
| Woofer | 40–300 Hz | Low Fs, moderate Xmax, high Qms |
| Midrange | 300–3000 Hz | Low distortion, controlled breakup, Fs < 100 Hz |
| Tweeter | 3000–20000 Hz | Extended response, low mass, wide dispersion |

</div>

## Candidate Driver Classes

### Woofer

- **Type**: 8–10" paper or aluminum cone
- **Candidates**: SB Acoustics, Scan-Speak, Seas
- **Key specs**: Fs < 35 Hz, Vas 40–80 L, Xmax > 8 mm

### Midrange

- **Type**: 5–6" paper or polypropylene cone
- **Candidates**: Scan-Speak Revelator/Illuminator, Seas Excel, Accuton
- **Key specs**: Fs < 60 Hz, controlled breakup > 4 kHz, low distortion

### Tweeter

- **Type**: Ribbon or AMT (Air Motion Transformer)
- **Candidates**: RAAL, Mundorf, Fountek
- **Key specs**: Low-end extension to 2.5–3 kHz, dispersion matching midrange

## Integration with Active DSP

Driver selection is less critical in an active system:

- **Response correction**: DSP compensates for response anomalies
- **Crossover flexibility**: Steep digital slopes protect drivers
- **Time alignment**: Per-driver delay compensates for acoustic offset
- **Linearization**: Can correct for mild nonlinearities

The primary driver selection criteria become:

1. **Power handling / thermal limits**: Cannot be DSP-corrected
2. **Distortion floor**: Fundamental limit on system performance
3. **Dispersion pattern**: Determines off-axis response, DSP has limited ability to correct

# Mono Bass Cabinet Architecture

## Performance Targets

<div class="center">

| **Parameter** | **Target** | **Notes** |
|:--------------|:-----------|:----------|
| –3 dB extension | 25 Hz | In-room |
| Maximum SPL | 110 dB @ 1m | 40–80 Hz band |
| THD at rated SPL | < 5% | Above port tuning |
| Power handling | 500 W continuous | Thermal-limited |

</div>

## Driver Configuration Options

### Single Large Driver

- **Driver**: 15–18" high-excursion woofer
- **Advantages**: Simplicity, single voice coil to drive
- **Disadvantages**: Large cabinet, limited high-SPL headroom

### Dual Opposed Drivers

- **Driver**: Two 12–15" woofers in push-push configuration
- **Advantages**: Force cancellation, reduced cabinet vibration
- **Disadvantages**: Increased cost, matching requirements

### Driver Array

- **Driver**: Four 10–12" woofers
- **Advantages**: Distributed excursion, thermal capacity
- **Disadvantages**: Increased complexity, cost

The **dual opposed** configuration offers the best balance of performance, cabinet size, and vibration cancellation for this application.

## Alignment Choices

### Sealed (Infinite Baffle)

- Simple, compact, excellent transient response
- –3 dB point limited by driver Vas and cabinet size
- Qtc ~0.7 for maximally flat response

### Vented (Bass Reflex)

- Extended low-frequency response
- Port tuning Fb typically 0.8× –3 dB target
- Unloading below Fb requires high-pass protection

### Hybrid (Sealed + Passive Radiator)

- Extended response without port noise
- Passive radiator tuned like vent but no air velocity
- Additional cost and alignment complexity

Recommendation: **Sealed alignment** for simplicity and transient response, sized for –3 dB at 28–30 Hz, with DSP extension to achieve 25 Hz target.

## Metamaterial Application in Bass Module

The bass module can employ metamaterial concepts for:

- **Upper-bass absorption**: Standing wave control in 80–200 Hz range
- **Port treatment**: If vented, serpentine port with absorption cells
- **Panel damping**: Helmholtz cells embedded in cabinet walls

However, the scale of required structures at bass frequencies (wavelength at 50 Hz = 7 m) limits the utility of metamaterial approaches. Primary cavity shaping focuses on standing wave disruption rather than broadband absorption.

# Cost Structure & Manufacturing Economics

## Target BOM Breakdown (Tower Pair)

<div class="center">

| **Category** | **Estimated Cost** | **Notes** |
|:-------------|:-------------------|:----------|
| Drivers (6 total) | $800–1,200 | High-end but not exotic |
| MDF material | $150–250 | ~40 sheets per pair |
| Cutting labor | $300–500 | CNC time, setup |
| Assembly labor | $400–600 | Stacking, bonding, finishing |
| Finishing materials | $150–300 | Hardener, lacquer, hardware |
| Amplification (6ch) | $400–800 | Class D modules |
| DSP/control | $200–400 | Integrated processor |
| Hardware/connectors | $100–200 | Binding posts, feet, etc. |
| **Total BOM** | **$2,500–4,250** | |

</div>

**Cost validation required**: The BOM estimate spans a 70% range. The "cutting labor" estimate ($300–500) requires validation against actual CNC/laser cutting service quotes for representative internal geometry. If actual costs are 2× higher than estimated, retail price positioning changes substantially. Recommend obtaining quotes for a 10-slice representative sample before finalizing BOM projections.

## Low-Volume Production Assumptions

- **Batch size**: 10–50 pairs per run
- **Lead time**: 4–8 weeks from order
- **Quality control**: 100% functional test, 10% acoustic measurement

## Retail Pricing Rationale

<div class="center">

| **Scenario** | **BOM** | **Multiplier** | **Retail (pair)** |
|:-------------|:--------|:---------------|:------------------|
| Conservative | $4,250 | 3× | $12,750 |
| Optimized | $2,500 | 3× | $7,500 |
| With margin for R&D | $3,500 | 3.5× | $12,250 |

</div>

Target retail: **$8,000–12,000 per pair** positions Strata in the upper tier of high-end loudspeakers, below exotic materials (diamond, beryllium) but competing with established audiophile brands on performance and differentiated on design/technology visibility.

# Risk Areas & Open Engineering Questions

## Port Noise and Outlet Geometry

**Severity: High**

The metaport design assumes distributed outlets will reduce air velocity and thus turbulence noise. This requires validation:

- **Measurement**: Noise floor with excitation sweeps
- **Threshold determination**: At what SPL does port noise become audible?
- **Geometry iteration**: May require multiple prototypes
- **Quantitative requirement**: See Port Noise Analysis in Exposed Metaport section

## Simulation-Reality Gap

**Severity: High**

FDTD models may not capture viscous/thermal losses in narrow channels; physical behavior may diverge from simulation:

- **Validation**: Simulation-to-measurement correlation required before relying on optimization
- **Loss models**: Empirical characterization of neck geometry loss factors
- **Mitigation**: Build simplified test geometries to validate loss predictions before full system simulation

## Manufacturing Tolerances

**Severity: Medium**

Acoustic metamaterial performance depends on dimensional accuracy:

- **Cavity volume tolerance**: ±5% affects resonant frequency by ~2.5%
- **Neck dimensions**: Small necks are sensitive to variations
- **Layer alignment**: Misalignment affects channel geometry

Mitigation: Design for robustness, include ±1 mm tolerance in simulation evaluation.

## Finish Repeatability

**Severity: Medium**

Exposed strata finishing requires consistent execution:

- **Edge hardening**: Process variability affects appearance
- **Clear coat application**: Runs, sags, orange peel
- **Color matching**: Batch-to-batch MDF variation

Mitigation: Develop finishing spec with quality gates; source MDF in batch quantities.

## Driver Variability

**Severity: Low**

Even high-quality drivers have unit-to-unit variation:

- **Sensitivity**: ±0.5 dB typical
- **Fs, Qts, Vas**: ±10–15% typical
- **Frequency response**: ±1.5 dB in-band

Mitigation: Active DSP with per-unit calibration. Measure each driver; store compensation in DSP.

## Panel Vibration and Vibro-Acoustic Modeling

**Severity: Medium**

This specification focuses on cavity acoustics. Panel vibration is a separate concern requiring attention:

- **Modal analysis**: Required to identify problem frequencies
- **Target specification**: First panel mode > 500 Hz, modal Q < 5
- **Damping strategy**: Constrained-layer, mass loading, bracing
- **Coupling**: Cavity modes may excite panel modes

Future work: Coupled vibro-acoustic simulation linking cavity acoustics to structural dynamics. Panel modes should be characterized in Phase 3 prototype measurements.

## Potential Failure Modes

Beyond the primary risks above:

- **Resonator mistuning due to humidity**: MDF expands/contracts with moisture; acoustic properties may vary seasonally
- **Glue line acoustic transmission**: Lamination joints may create unintended acoustic paths
- **Thermal drift**: Temperature-dependent air density affects Helmholtz tuning

# Development Roadmap

## Phase 1: Simulation Foundation

- [ ] Develop FDTD simulation framework
- [ ] Validate against known geometries (transmission line, sealed box)
- [ ] Define parametric geometry families for optimization
- [ ] Establish objective metrics and thresholds

## Phase 2: Midrange Metachamber Development

**This phase is the minimal viable validation milestone** that can confirm or falsify the core hypothesis before committing to full system development.

- [ ] Design initial metachamber geometry
- [ ] Optimize via simulation
- [ ] Build sub-scale test chamber (half-size prototype)
- [ ] Measure and correlate with simulation
- [ ] **Go/no-go decision**: If simulation-measurement correlation < 80% or stored energy reduction < 50% vs. stuffed reference, revisit approach

## Phase 3: Full-Scale Midrange Prototype

- [ ] Design complete midrange cavity
- [ ] Select candidate midrange driver
- [ ] Build full-scale single-driver prototype
- [ ] Measure impulse response, decay, frequency response
- [ ] Measure panel modes and structural dynamics
- [ ] **Baseline comparison**: Build equivalent stuffed enclosure for A/B measurement

## Phase 4: Tower Integration

- [ ] Complete tower design with all three drivers
- [ ] Build first complete tower prototype
- [ ] Develop DSP crossover and calibration
- [ ] Acoustic measurement: anechoic + in-room

## Phase 5: Bass Module and System Integration

- [ ] Design bass module
- [ ] Build bass module prototype
- [ ] Integrate tower + bass system
- [ ] System tuning and optimization

## Phase 6: Production Validation

- [ ] Finalize manufacturing spec
- [ ] Validate BOM with actual fabrication quotes
- [ ] Build production-intent prototypes
- [ ] Qualify finishing process
- [ ] Listening evaluation and refinement

# Why This Is Hard to Copy

The defensibility of Strata comes from **integration**, not any single trick. The system comprises multiple interlocking innovations that create value only when combined:

<div class="center">

| **Element** | **Alone** | **In System** |
|:------------|:----------|:--------------|
| Geometry-dependent acoustics | Requires simulation to design | Enables predictable, repeatable behavior |
| Laminated MDF construction | Just a manufacturing method | Enables complex 3D geometry at low cost |
| Simulation-driven optimization | Requires geometry to simulate | Produces validated designs before cutting |
| Active electronics + DSP | Standard audiophile feature | Eliminates passive crossovers, enables calibration |
| Aesthetic exposure of function | Cosmetic decision | Differentiates and communicates technology |

</div>

**Copying one element without the others yields little benefit:**

- Laminated MDF without simulation-driven geometry is just an expensive box
- Exposed cutaways without functional acoustic design are decorative fraud
- Metamaterial cavities without active electronics can't be properly aligned
- Simulation-driven design without manufacturable geometry is academic exercise

The integrated system creates:

- **Technical moat**: Multi-disciplinary expertise requirement (acoustics + manufacturing + DSP + industrial design)
- **IP protection**: The specific geometry families and optimization methods are proprietary
- **Brand identity**: The visual language is instantly recognizable and tied to function
- **Barrier to entry**: Competitors must recreate the entire development pipeline, not just copy a finished design

This integration-based defensibility is stronger than any single patent because it requires simultaneous capability across multiple domains.

# References

<div class="thebibliography">

E. Choi and W. Jeon, *Near-perfect sound absorption using hybrid resonance between subwavelength Helmholtz resonators with non-uniformly partitioned cavities*, Scientific Reports, vol. 14, no. 3156, 2024. https://doi.org/10.1038/s41598-024-53595-y

</div>

# Appendices (Future)

The following appendices will be developed as the project progresses:

## A. Simulation Parameter Tables

- Material properties (MDF, air)
- Boundary conditions
- Mesh specifications
- Viscous/thermal loss models for narrow channels

## B. Driver Electrical/Mechanical Data

- T/S parameters for selected drivers
- Impedance curves
- Power handling specifications

## C. CAD / Slice Generation Notes

- Software toolchain
- Nesting strategy
- Registration feature design

## D. Measurement Methodology

- Microphone positions
- Calibration procedures
- Post-processing algorithms
- Baseline comparison protocol (Strata vs. conventional stuffed enclosure)

## E. Listening Test Protocol

- Panel selection
- Test material
- Evaluation criteria
- ABX testing procedure for coloration detection