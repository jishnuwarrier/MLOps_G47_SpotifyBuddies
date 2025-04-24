from pydantic import BaseModel

# TODO => Modify when input are more defined


class PlaylistRequestSchema(BaseModel):
    user_id: str


class PlaylistResponseSchema(BaseModel):
    playlist_id: str
