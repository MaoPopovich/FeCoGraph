import time
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread
import logging

logger = logging.getLogger(__name__)

class FedAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(args, clientAVG)

        # Declare join ratio and total clients
        logger.info("Join ratio / total clients: {:.2f} / {:d}".format(self.join_ratio, self.num_clients))
        logger.info("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []


    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                logger.info(f"-------------Round number: {i}-------------")
                logger.info("Evaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            logger.info('-'*25 + 'time cost' + '-'*25 + str(self.Budget[-1]))

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        logger.info("Best accuracy." + str(max(self.rs_test_acc)))
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        logger.info("Average time cost per round." + str(sum(self.Budget[1:])/len(self.Budget[1:])))

        self.save_results()
        self.save_global_model()
