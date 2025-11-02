from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread
import logging

logger = logging.getLogger(__name__)

class Local(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(args, clientAVG)

        logger.info("Join ratio / total clients: {:.2f} / {:d}".format(self.join_ratio, self.num_clients))
        logger.info("Finished creating server and clients.")
        
        # self.load_model()


    def train(self):
        for i in range(self.global_rounds+1):
            self.selected_clients = self.select_clients()

            if i%self.eval_gap == 0:
                logger.info(f"-------------Round number: {i}-------------")
                logger.info("Evaluate personalized models")
                self.evaluate()

            self.selected_clients = self.select_clients()
            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break


        logger.info("Best accuracy."+str(max(self.rs_test_acc)))
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))

        self.save_results()
        self.save_global_model()
