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
    from . import ppo_gated  # type: ignore
    AGENTS['ppo'] = ppo_gated
except Exception as _e:
    pass

try:
    from . import envelope_agent  # type: ignore
    AGENTS['envelope'] = envelope_agent
except Exception as _e:
    pass

try:
    from . import pgmorl_agent  # type: ignore
    AGENTS['pgmorl'] = pgmorl_agent
except Exception as _e:
    pass

try:
    from . import sec_pcn_agent  # type: ignore
    AGENTS['sec-pcn'] = sec_pcn_agent
except Exception as _e:
    pass



