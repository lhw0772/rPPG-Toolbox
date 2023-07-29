import transforms
import torch

def apply_transformations(self, clip, augment=True):
    speed = 1.0
    if augment:
        ## Time resampling
        if self.aug_speed:
            clip, speed = transforms.augment_speed(clip, self.frames_per_clip,
                                                   self.channels, self.speed_slow, self.speed_fast)

        ## Randomly horizontal flip
        if self.aug_flip:
            clip = transforms.augment_horizontal_flip(clip)

        ## Randomly reverse time
        if self.aug_reverse:
            clip = transforms.augment_time_reversal(clip)

        ## Illumination noise
        if self.aug_illum:
            clip = transforms.augment_illumination_noise(clip)

        ## Gaussian noise for every pixel
        if self.aug_gauss:
            clip = transforms.augment_gaussian_noise(clip)

        ## Random resized cropping
        if self.aug_resizedcrop:
            clip = transforms.random_resized_crop(clip)

    clip = torch.from_numpy(clip.copy()).float()

    return clip, speed
