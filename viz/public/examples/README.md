# FDTD Examples Gallery

This directory contains curated example simulations for the FDTD acoustic strata_fdtd simulator. These examples demonstrate fundamental concepts, common use cases, and best practices.

## Directory Structure

```
examples/
├── index.json              # Gallery metadata (used by web UI)
├── README.md              # This file
├── *.py                   # Example Python scripts
├── *.svg                  # Thumbnail images for gallery
└── template.py            # Template for creating new examples
```

## Available Examples

### Basics

1. **basic-pulse.py** - Gaussian pulse propagation
   - Grid: 100 × 100 × 100 cells
   - Runtime: ~1 minute
   - Concepts: Uniform grid, single source, time-domain probes, PML boundaries

2. **resonant-cavity.py** - Standing waves in rectangular cavity
   - Grid: 100 × 100 × 150 cells
   - Runtime: ~2 minutes
   - Concepts: Rigid boundaries, resonance modes, fundamental frequency

### Physics Demonstrations

3. **doppler-effect.py** - Moving source frequency shift
   - Grid: 150 × 100 × 100 cells
   - Runtime: ~2 minutes
   - Concepts: Moving sources, Doppler shift, time-domain analysis

4. **interference.py** - Two-source interference pattern
   - Grid: 150 × 150 × 100 cells
   - Runtime: ~2 minutes
   - Concepts: Coherent sources, interference fringes, phase relationships

### Transducers

5. **pzt-transducer.py** - Piezoelectric ultrasonic transducer
   - Grid: 150 × 150 × 200 cells
   - Runtime: ~3 minutes
   - Concepts: PZT materials, sinusoidal excitation, radiation patterns

### Metamaterials

6. **acoustic-lens.py** - Graded-index focusing lens
   - Grid: 200 × 200 × 150 cells
   - Runtime: ~3 minutes
   - Concepts: Graded materials, plane waves, beam focusing

### Advanced Topics

7. **phased-array.py** - Beam steering with phased array
   - Grid: 200 × 150 × 200 cells
   - Runtime: ~4 minutes
   - Concepts: Multi-element arrays, time delays, beam forming

8. **nonuniform-grid.py** - Variable resolution grid
   - Grid: 100 × 100 × 200 cells (variable spacing)
   - Runtime: ~2 minutes
   - Concepts: Adaptive grids, memory optimization, CFL stability

## Using Examples

### From the Web UI

1. Open the Examples Gallery in the web UI
2. Browse examples by category
3. Click "View Code" to see the full script
4. Click "Load" to populate the Builder Mode editor
5. Run the simulation

### From the Command Line

```bash
# Run example directly
fdtd-compute examples/basic-pulse.py

# Copy example to customize
cp examples/basic-pulse.py my-simulation.py
# Edit my-simulation.py
fdtd-compute my-simulation.py
```

## Creating New Examples

Use the provided template as a starting point:

```bash
cp examples/template.py examples/my-example.py
```

### Example Script Guidelines

**Structure:**
1. **Docstring** - Clear title and description of concepts demonstrated
2. **Imports** - Standard strata_fdtd imports
3. **Grid setup** - Document grid size and resolution choices
4. **Scene configuration** - Add sources, materials, probes with comments
5. **Solver setup** - Configure duration, boundary conditions
6. **Info output** - Print simulation parameters for user reference

**Best Practices:**
- Keep runtime under 5 minutes on a modern laptop
- Include informative comments explaining why, not just what
- Print key simulation parameters (grid size, timestep, estimated runtime)
- Use realistic physical parameters
- Add multiple probes to capture interesting behavior
- Document expected results in comments

**Code Quality:**
- Follow PEP 8 style guidelines
- Use descriptive variable names
- Add type hints where helpful
- Keep scripts self-contained (no external dependencies)

### Adding to Gallery

To add your example to the web UI gallery:

1. **Create the Python script** in `web-ui/public/examples/`

2. **Create a thumbnail** (400×300 px SVG recommended):
   ```bash
   # Name it same as script but with .svg extension
   my-example.svg
   ```

3. **Add metadata** to `index.json`:
   ```json
   {
     "id": "my-example",
     "title": "My Example Title",
     "description": "Brief description (1-2 sentences)",
     "file": "my-example.py",
     "thumbnail": "my-example.svg",
     "category": "Basics|Physics|Transducers|Metamaterials|Imaging|Advanced",
     "difficulty": "Beginner|Intermediate|Advanced",
     "tags": ["tag1", "tag2", "tag3"],
     "estimatedRuntime": "~X min",
     "estimatedSize": "XX MB",
     "gridSize": "NX × NY × NZ",
     "features": [
       "Feature 1",
       "Feature 2",
       "Feature 3",
       "Feature 4"
     ]
   }
   ```

4. **Test the example**:
   ```bash
   # Verify it runs without errors
   fdtd-compute examples/my-example.py

   # Check gallery displays correctly
   # (load web UI and navigate to Examples Gallery)
   ```

## Runtime and Size Optimization

**Reduce Runtime:**
- Use coarser grid resolution (but maintain at least 10-15 cells per wavelength)
- Shorter simulation duration
- Smaller grid size
- Use nonuniform grids for selective refinement
- Enable native C++ backend if available

**Reduce Output Size:**
- Fewer probes
- Lower probe sampling rate
- Shorter duration
- Smaller grid

**Example Grid Resolution Guidelines:**

| Frequency | Wavelength (air) | Recommended Resolution | Min Cells/λ |
|-----------|------------------|------------------------|-------------|
| 1 kHz     | 343 mm           | 20-30 mm               | 11-17       |
| 10 kHz    | 34.3 mm          | 2-3 mm                 | 11-17       |
| 40 kHz    | 8.6 mm           | 0.5-0.8 mm             | 11-17       |
| 1 MHz     | 0.34 mm          | 20-30 μm               | 11-17       |

## Thumbnail Creation

Thumbnails should visually represent the simulation concept. Guidelines:

**Technical Requirements:**
- Format: SVG (scalable, no dependencies) or PNG (400×300 px)
- File size: < 100 KB
- Filename: Same as script with `.svg` or `.png` extension

**Visual Design:**
- Use gradient backgrounds matching the category color theme
- Include visual elements representing the physics (waves, sources, etc.)
- Add title text at bottom with semi-transparent background
- Use colors from the category palette (defined in index.json)

**Category Colors:**
- Basics: Blue (#2196f3)
- Physics: Red/Cyan (#f44336, #00bcd4)
- Transducers: Green (#4caf50)
- Metamaterials: Purple (#9c27b0)
- Imaging: Indigo (#3f51b5)
- Advanced: Brown (#795548)

## Validation Checklist

Before submitting a new example:

- [ ] Script runs without errors
- [ ] Runtime is under 5 minutes on reference hardware
- [ ] Output size is documented accurately
- [ ] Code follows style guidelines
- [ ] Comments explain the physics and design choices
- [ ] Docstring includes clear title and description
- [ ] Prints helpful simulation info before running
- [ ] Metadata added to index.json
- [ ] Thumbnail created and properly sized
- [ ] Features list highlights key concepts (4 items)
- [ ] Tags are relevant and consistent with other examples
- [ ] Category and difficulty are appropriate

## Reference Hardware

Runtime estimates assume:
- CPU: Modern laptop (Apple M1 or Intel Core i7 equivalent)
- RAM: 16 GB
- Backend: Native C++ kernels if available, otherwise Python

## Contributing

To contribute new examples:

1. Fork the repository
2. Create example following guidelines above
3. Test thoroughly
4. Submit pull request with:
   - The example script
   - Thumbnail image
   - Updated index.json
   - Description of what the example demonstrates

## Support

For questions about examples:
- Open an issue on GitHub
- Tag with `examples` and `documentation` labels
- Provide details about which example and what you're trying to do

## License

All examples are provided under the same license as the main project.
See the repository LICENSE file for details.

---

**Last Updated:** 2024-12-14
**Total Examples:** 8
**Categories:** 6 (Basics, Physics, Transducers, Metamaterials, Imaging, Advanced)
