from pydantic import BaseModel

# TODO => Modify when input are more defined


class PlaylistRequestSchema(BaseModel):
    user_ids: list[int]


class PlaylistResponseSchema(BaseModel):
    user_id: int
    playlists: list[int]
