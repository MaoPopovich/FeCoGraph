import copy
import numpy as np
import time
from flcore.clients.clientditto import clientDitto
from flcore.servers.serverbase import Server
from threading import Thread
import os
import h5py
import logging

logger = logging.getLogger(__name__)

class Ditto(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(args, clientDitto)

        logger.info("Join ratio / total clients: {:.2f} / {:d}".format(self.join_ratio, self.num_clients))
        logger.info("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.rs_train_acc_per = []
        self.rs_train_loss_per = []
        self.rs_test_acc_per = []
        self.rs_acc_std_per = []


    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            # if i%self.eval_gap == 0:
            #     print(f"\n-------------Round number: {i}-------------")
            #     print("\nEvaluate global models")
            #     self.evaluate()

            for client in self.selected_clients:
                client.ptrain()  # regularization between personalized model and global model
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            if i%self.eval_gap == 0:
                logger.info(f"-------------Round number: {i}-------------")
                logger.info("Evaluate personalized models")
                self.evaluate_personalized()

            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            logger.info('-'*25 + 'time cost' + '-'*25 + str(self.Budget[-1]))

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        logger.info("Best accuracy." + str(max(self.rs_test_acc_per)))
        logger.info("Average time cost per round." + str(sum(self.Budget[1:])/len(self.Budget[1:])))

        self.save_results()
        self.save_global_model()


    def test_metrics_personalized(self):
        num_samples = []
        tot_correct = []

        for c in self.clients:
            ct, ns = c.test_metrics_personalized()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct

    def train_metrics_personalized(self):
        num_samples = []
        losses = []
        tot_correct = []
        for c in self.clients:
            ct, cl, ns = c.train_metrics_personalized()
            num_samples.append(ns)
            losses.append(cl*1.0)
            tot_correct.append(ct*1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, losses

    # evaluate selected clients
    def evaluate_personalized(self):
        stats = self.test_metrics_personalized()
        stats_train = self.train_metrics_personalized()

        test_acc = (sum(stats[2])*1.0 / sum(stats[1])).cpu()
        train_acc = (sum(stats_train[2])*1.0 / sum(stats_train[1])).cpu()
        train_loss = (sum(stats_train[3])*1.0 / sum(stats_train[1])).cpu()
        accs = [(a / n).cpu() for a, n in zip(stats[2], stats[1])]
        
        self.rs_test_acc_per.append(test_acc)
        self.rs_train_acc_per.append(train_acc)
        self.rs_train_loss_per.append(train_loss)
        self.rs_acc_std_per.append(np.std(accs))

        logger.info("Averaged Train Loss: {:.4f}".format(train_loss))
        logger.info("Averaged Train Accuracy: {:.4f}".format(train_acc))
        logger.info("Averaged Test Accurancy: {:.4f}".format(test_acc))
        logger.info("Std Test Accurancy: {:.4f}".format(np.std(accs)))

    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = self.outdir
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.rs_test_acc_per)):
            algo = algo + "_" + self.goal + "_lr" + str(self.learning_rate) + "_rs" + str(self.global_rounds) + "_ls" + str(self.local_epochs)
            file_path = result_path + '/' + "{}.h5".format(algo)
            logger.info("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc_per)
                hf.create_dataset('rs_acc_std', data=self.rs_acc_std_per)
                hf.create_dataset('rs_train_acc', data=self.rs_train_acc_per)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss_per)