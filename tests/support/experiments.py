"""Shared authored experiment fixtures."""


def declaration():
    return {
        "name": "particles",
        "case": "tiny.json",
        "base": {"hardware": "local"},
        "factors": ["num_particles"],
        "arms": [
            {"name": "small", "settings": {"num_particles": 8}},
            {"name": "large", "settings": {"num_particles": 16}},
        ],
        "seeds": [{"generation": 9, "sampler": 13}],
    }
