#from augment import EffectChain

from fairseq.data import BaseWrapperDataset
from fairseq.data.audio.speech_to_text_dataset import (
    SpeechToTextDataset,
    SpeechToTextDatasetItem,
)
import numpy as np
from typing import Union, List, Dict
import torch
import torch.nn.functional as F

SAMPLING_RATE = 16000
ECHO_GAININ = 0.8
ECHO_GAINOUT = 0.9


class SpeechAugmentationDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset: SpeechToTextDataset,
        effects_info: Dict[str, Union[List, Dict]],
        p_augm: float,
        max_src_len: int
    ) -> None:       

        super(SpeechAugmentationDataset, self).__init__(dataset)

        self.effects_info = effects_info
        self.p_augm = p_augm
        self.max_src_len = max_src_len

        self.sr = SAMPLING_RATE
        self.echo_gainin = ECHO_GAININ
        self.echo_gainout = ECHO_GAINOUT
        self.info = {"rate": self.sr}

        # add effects
        self.avai_effects = []
        if self.effects_info["tempo"][0] != self.effects_info["tempo"][1]:
            self.avai_effects.append("tempo")
        if self.effects_info["pitch"][0] != self.effects_info["pitch"][1]:
            self.avai_effects.append("pitch")
        cond_delay = (
            self.effects_info["echo"]["delay"][0]
            != self.effects_info["echo"]["delay"][1]
        )
        cond_decay = (
            self.effects_info["echo"]["decay"][0]
            != self.effects_info["echo"]["decay"][1]
        )
        if cond_delay and cond_decay:
            self.avai_effects.append("echo")

    def __getitem__(self, index: int) -> SpeechToTextDatasetItem:
        example = self.dataset.__getitem__(index)
        
        # apply effects or keep the original audiowave
        if np.random.rand() < self.p_augm:
            example.source = self._augment(example.source)

        # normalize audiowave
        example.source = self._normalize(example.source)

        return example

    def _augment(self, input_tensor: torch.tensor) -> torch.tensor:

        src_len = len(input_tensor)

        # init empty chain
        effect_chain = EffectChain()

        if "pitch" in self.avai_effects:

            effect_chain = effect_chain.pitch(
                np.random.randint(*self.effects_info["pitch"])
            ).rate(self.sr)

        if "tempo" in self.avai_effects:

            # don't apply a tempo effect that will create an example longer
            # than max_src_len
            min_tempo_value = src_len / self.max_src_len
            sampled_tempo_value = np.random.uniform(
                max(min_tempo_value, self.effects_info["tempo"][0]),
                self.effects_info["tempo"][1],
            )
            effect_chain = effect_chain.tempo(sampled_tempo_value)

            # adjust to length after tempo
            src_len = int(src_len * sampled_tempo_value)

        if "echo" in self.avai_effects:

            effect_chain = effect_chain.echo(
                self.echo_gainin,
                self.echo_gainout,
                np.random.randint(*self.effects_info["echo"]["delay"]),
                np.random.uniform(*self.effects_info["echo"]["decay"]),
            )

        # apply effects and reduce channel dimension that is created by default
        # also crop the extra frames that were created by echo delay
        input_tensor_augm = effect_chain.apply(
            input_tensor, src_info=self.info, target_info=self.info
        ).squeeze(0)[:src_len]

        # sox might misbehave sometimes by giving nan/inf if sequences are too short
        # (or silent)
        if torch.isnan(input_tensor_augm).any() or torch.isinf(input_tensor_augm).any():
            print("Warning: Augmentation failed.")
            return input_tensor
        else:
            return input_tensor_augm

    def _normalize(self, input_tensor: torch.tensor) -> torch.tensor:
        with torch.no_grad():
            input_tensor = F.layer_norm(input_tensor, input_tensor.shape)
        return input_tensor

    def worker_init_fn(worker_id: int) -> None:
        np.random.seed(np.random.get_state()[1][0] + worker_id)
