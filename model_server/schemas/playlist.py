from pydantic import BaseModel

# TODO => Modify when input are more defined


class PlaylistRequestSchema(BaseModel):
    user_id: int


class PlaylistResponseSchema(BaseModel):
    playlist_id: int
