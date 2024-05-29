import torch
from torch.nn import functional as F

from dowhy.causal_prediction.algorithms.base_algorithm import PredictionAlgorithm
from dowhy.causal_prediction.algorithms.regularization import Regularizer


class CACM(PredictionAlgorithm):
    def __init__(
        self,
        model,
        sequence_classification=False,
        optimizer="Adam",
        lr=1e-3,
        weight_decay=0.0,
        betas=(0.9, 0.999),
        momentum=0.9,
        kernel_type="gaussian",
        ci_test="mmd",
        attr_types=[],
        E_conditioned=True,
        E_eq_A=[],
        gamma=1e-6,
        lambda_causal=1.0,
        lambda_conf=1.0,
        lambda_ind=1.0,
        lambda_sel=1.0,
    ):
        
        super().__init__(model, optimizer, lr, weight_decay, betas, momentum)

        self.CACMRegularizer = Regularizer(E_conditioned, ci_test, kernel_type, gamma)

        self.attr_types = attr_types
        self.E_eq_A = E_eq_A
        self.lambda_causal = lambda_causal
        self.lambda_conf = lambda_conf
        self.lambda_ind = lambda_ind
        self.lambda_sel = lambda_sel

        self.sequence_classification = sequence_classification

    def training_step(self, train_batch, batch_idx):

        minibatches = train_batch

        objective = 0
        correct, total = 0, 0
        penalty_causal, penalty_conf, penalty_ind, penalty_sel = 0, 0, 0, 0
        nmb = len(minibatches)

        if not self.sequence_classification:
            self.featurizer = self.model[0]
            self.classifier = self.model[1]

            features = [self.featurizer(xi) for xi, _, _ in minibatches]
            classifs = [self.classifier(fi) for fi in features]
            targets = [yi for _, yi, _ in minibatches]

        else:
            self.classifier = self.model

            features = train_batch[0]
            targets = train_batch[1]

            classifs = [self.classifier(fi) for fi in features]


        for i in range(nmb):
            objective += F.cross_entropy(classifs[i], targets[i])
            correct += (torch.argmax(classifs[i], dim=1) == targets[i]).float().sum().item()
            total += classifs[i].shape[0]

        objective /= nmb
        loss = objective

        if self.attr_types != []:
            for attr_type_idx, attr_type in enumerate(self.attr_types):
                attribute_labels = [
                    ai for _, _, ai in minibatches
                ]

                E_eq_A_attr = attr_type_idx in self.E_eq_A

                # Acause regularization
                if attr_type == "causal":
                    penalty_causal += self.CACMRegularizer.conditional_reg(
                        classifs, [a[:, attr_type_idx] for a in attribute_labels], [targets], nmb, E_eq_A_attr
                    )

                # Aconf regularization
                elif attr_type == "conf":
                    penalty_conf += self.CACMRegularizer.unconditional_reg(
                        classifs, [a[:, attr_type_idx] for a in attribute_labels], nmb, E_eq_A_attr
                    )

                # Aind regularization
                elif attr_type == "ind":
                    penalty_ind += self.CACMRegularizer.unconditional_reg(
                        classifs, [a[:, attr_type_idx] for a in attribute_labels], nmb, E_eq_A_attr
                    )

                # Asel regularization
                elif attr_type == "sel":
                    penalty_sel += self.CACMRegularizer.conditional_reg(
                        classifs, [a[:, attr_type_idx] for a in attribute_labels], [targets], nmb, E_eq_A_attr
                    )

            if nmb > 1:
                penalty_causal /= nmb * (nmb - 1) / 2
                penalty_conf /= nmb * (nmb - 1) / 2
                penalty_ind /= nmb * (nmb - 1) / 2
                penalty_sel /= nmb * (nmb - 1) / 2

            # Compile loss
            loss += self.lambda_causal * penalty_causal
            loss += self.lambda_conf * penalty_conf
            loss += self.lambda_ind * penalty_ind
            loss += self.lambda_sel * penalty_sel

            if torch.is_tensor(penalty_causal):
                penalty_causal = penalty_causal.item()
                self.log("penalty_causal", penalty_causal, on_step=False, on_epoch=True, prog_bar=True)
            if torch.is_tensor(penalty_conf):
                penalty_conf = penalty_conf.item()
                self.log("penalty_conf", penalty_conf, on_step=False, on_epoch=True, prog_bar=True)
            if torch.is_tensor(penalty_ind):
                penalty_ind = penalty_ind.item()
                self.log("penalty_ind", penalty_ind, on_step=False, on_epoch=True, prog_bar=True)
            if torch.is_tensor(penalty_sel):
                penalty_sel = penalty_sel.item()
                self.log("penalty_sel", penalty_sel, on_step=False, on_epoch=True, prog_bar=True)

        elif self.graph is not None:
            pass  # TODO

        else:
            raise ValueError("No attribute types or graph provided.")

        acc = correct / total

        metrics = {"train_acc": acc, "train_loss": loss}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)

        return loss
