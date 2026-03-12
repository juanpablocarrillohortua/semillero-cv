# ORBIX Semillero — Claude Code

Lee `AGENTS.md` para tu identidad completa, entorno, stack y reglas.

## Routing de Skills

Cuando el usuario pida una tarea de CV, lee el skill correspondiente antes de generar codigo:

| Tarea | Archivo |
|-------|---------|
| Deteccion de objetos | `skills/object-detection.md` |
| Pose estimation | `skills/pose-estimation.md` |
| Tracking | `skills/tracking.md` |
| Segmentacion | `skills/segmentation.md` |
| Pipeline de video | `skills/video-pipeline.md` |
| Entrenar modelo | `skills/training.md` |

## Reglas

- Verificar GPU antes de entrenar: `nvidia-smi`
- Usar `templates/project/` para scaffolding de proyectos nuevos
- Leer `knowledge/environment.md` si necesitas specs del hardware
- Leer `knowledge/frameworks.md` si necesitas referencia de una libreria
- Codigo siempre con `device="cuda"` y fallback a CPU
- No crear archivos innecesarios — mantener el repo limpio
- Datos en `data/` (gitignored), resultados en `outputs/` (gitignored)
