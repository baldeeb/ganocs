from .vanilla import            Discriminator
from .contextual import         (DepthAwareDiscriminator, 
                                 ContextAwareDiscriminator)
from .optimizer_wrapper import  (DiscriminatorWithOptimizer,
                                 DiscriminatorWithWessersteinOptimizer)
from .multi import              (MultiDiscriminatorWithOptimizer,
                                 MultiClassDiscriminatorWithOptimizer,
                                 MultiClassDiscriminatorWithWessersteinOptimizer,
                                 MultiDiscriminatorWithWessersteinOptimizer)
from .rgbd import               RgbdMultiDiscriminatorWithOptimizer
from .util import               get_multiple_discriminators