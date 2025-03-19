import asyncio
from viam.module.module import Module
from .models.scene_iq import SceneIq


if __name__ == '__main__':
    asyncio.run(Module.run_from_registry())
