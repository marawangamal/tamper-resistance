# Known bugs

In Axolotl YAML files, `message_property_mappings` properties must be strings.

# Storage

Do not create environments or caches under home. Use `$SCRATCH` for persistent
storage and `$SLURM_TMPDIR` for per-job environments.
