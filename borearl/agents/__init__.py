AGENTS = {}

# Import each agent module defensively so missing upstream deps don't crash the app
try:
    from . import eupg_agent  # type: ignore
    AGENTS['eupg'] = eupg_agent
except Exception as _e:
    pass

try:
    from . import pcn_agent  # type: ignore
    AGENTS['pcn'] = pcn_agent
except Exception as _e:
    pass

try:
    from . import chm_agent  # type: ignore
    AGENTS['chm'] = chm_agent
except Exception as _e:
    # CHM might not be available in some morl-baselines versions
    pass

try:
    from . import gpi_ls_agent  # type: ignore
    AGENTS['gpi_ls'] = gpi_ls_agent
except Exception as _e:
    pass



