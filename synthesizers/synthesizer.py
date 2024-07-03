from Batch import Batch
from FederatedTask import FederatedTask
from Params import Params


# The class to generate batches
class Synthesizer:
    params: Params
    task: FederatedTask

    def __init__(self, task: FederatedTask):
        self.task = task
        self.params = task.params

    def make_backdoor_batch(self, batch: Batch, test=False, attack=True) -> Batch:
        # 在数据批次中创建一个带有后门的副本，并返回这个修改后的批次

        # Don't attack if only normal loss task.
        # 判断是否需要攻击
        if (not attack) or (self.params.loss_tasks == ['normal'] and not test):
            return batch

        # 计算需要攻击的部分的大小
        if test:
            attack_portion = batch.batch_size
        else:
            attack_portion = round(batch.batch_size * self.params.poisoning_proportion)

        backdoored_batch = batch.clone()
        # change data
        self.apply_backdoor(backdoored_batch, attack_portion)

        return backdoored_batch

    def apply_backdoor(self, batch, attack_portion):
        """
        Modifies only a portion of the batch (represents batch poisoning).

        :param batch:
        :return:
        """
        self.synthesize_inputs(batch=batch, attack_portion=attack_portion)
        self.synthesize_labels(batch=batch, attack_portion=attack_portion)

        return

    def synthesize_inputs(self, batch, attack_portion=None):
        raise NotImplemented

    def synthesize_labels(self, batch, attack_portion=None):
        raise NotImplemented
