from enum import Enum
from pydantic import BaseModel

class Gender(Enum):
    FEMALE = "female"
    MALE = "male"
   
class Age(Enum):
    ADULT = "adult"
    YOUNG = "young"

class Pale_Skin(Enum):
    PALE = "pale"
    NOT_PALE = "not_pale"

class Wearing_hat(Enum):
    TRUE = "true"
    FALSE = "false"

class Goatee(Enum):
    TRUE = "true"
    FALSE = "false"

class Beard(Enum):
    TRUE = "true"
    FALSE = "false"

class Smiling(Enum):
    TRUE = "true"
    FALSE = "false"

class Eye_glasses(Enum):
    TRUE = "true"
    FALSE = "false"

class Bald(Enum):
    TRUE = "true"
    FALSE = "false"

class ODDImage(BaseModel):
    """
    Structured output format for GPT-based image metadata extraction
    Corresponds to CelebA dataset attributes
    """
    Gender: Gender
    Age: Age  
    Pale_Skin: Pale_Skin
    Wearing_hat: Wearing_hat  
    Goatee: Goatee
    Beard: Beard
    Smiling: Smiling
    Eye_glasses: Eye_glasses
    Bald: Bald