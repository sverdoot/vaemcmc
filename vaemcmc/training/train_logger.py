"""
Implementation of the Logger object for performing training logging and visualisation.
"""
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import utils as vutils
# from torch_mimicry.training import Logger



class TrainLogger:
    """
    Writes summaries and visualises training progress.
    
    Attributes:
        log_dir (str): The path to store logging information.
        num_steps (int): Total number of training iterations.
        dataset_size (int): The number of examples in the dataset.
        device (Device): Torch device object to send data to.
        flush_secs (int): Number of seconds before flushing summaries to disk.
        writers (dict): A dictionary of tensorboard writers with keys as metric names.
        num_epochs (int): The number of epochs, for extra information.
    """
    def __init__(
        self,
        log_dir,
        num_steps,
        dataset_size,
        device,
        flush_secs=120,
        **kwargs,
    ):
        # super().__init__(log_dir, num_steps, dataset_size, device, flush_secs, **kwargs)
        Path(log_dir).mkdir(exist_ok=True)
        self.log_dir = log_dir
        self.num_steps = num_steps
        self.dataset_size = dataset_size
        self.flush_secs = flush_secs
        self.num_epochs = self._get_epoch(num_steps)
        self.device = device
        self.writer = self._build_writer()
        self.writers = None
            
    def _get_epoch(self, steps):
        """
        Helper function for getting epoch.
        """
        return max(int(steps / self.dataset_size), 1)

    def _build_writer(self):
        writer = SummaryWriter(
            log_dir=Path(self.log_dir,),
            flush_secs=self.flush_secs,
        )

        return writer

    def write_summaries(self, log_data, global_step):
        """
        Tasks appropriate writers to write the summaries in tensorboard. Creates a
        dditional writers for summary writing if there are new scalars to log in
        log_data.

        Args:
            log_data (MetricLog): Dict-like object to collect log data for TB writing.
            global_step (int): Global step variable for syncing logs.

        Returns:
            None
        """
        for metric, data in log_data.items():
            #     if metric not in self.writers:
            #         self.writers[metric] = self._build_writer(metric)

            # Write with a group name if it exists
            # name = log_data.get_group_name(metric) or metric
            self.writer.add_scalar(
                metric,
                log_data[metric],
                global_step=global_step,  # name,
            )

    def close_writers(self):
        """
        Closes all writers.
        """
        # for metric in self.writers:
        #     self.writers[metric].close()
        self.writer.close()

    def print_log(self, global_step, log_data, time_taken):
        """
        Formats the string to print to stdout based on training information.

        Args:
            log_data (MetricLog): Dict-like object to collect log data for TB writing.
            global_step (int): Global step variable for syncing logs.
            time_taken (float): Time taken for one training iteration.

        Returns:
            str: String to be printed to stdout.
        """
        # Basic information
        log_to_show = [
            "INFO: [Epoch {:d}/{:d}][Global Step: {:d}/{:d}]".format(
                self._get_epoch(global_step), self.num_epochs, global_step,
                self.num_steps)
        ]

        # Display GAN information as fed from user.
        info = [""]
        metrics = sorted(log_data.keys())

        for metric in metrics:
            info.append('{}: {}'.format(metric, log_data[metric]))

        # Add train step time information
        info.append("({:.4f} sec/idx)".format(time_taken))

        # Accumulate to log
        log_to_show.append("\n| ".join(info))

        # Finally print the output
        ret = " ".join(log_to_show)
        print(ret)

        return ret

    def vis_images(self, model, global_step, num_images=64):
        """
        Produce visualisations of the G(z), one fixed and one random.

        Args:
            netG (Module): Generator model object for producing images.
            global_step (int): Global step variable for syncing logs.
            num_images (int): The number of images to visualise.

        Returns:
            None
        """
        img_dir = Path(self.log_dir, "images")
        Path(img_dir).mkdir(exist_ok=True)

        with torch.no_grad():
            # Generate random images
            images = model.sample((num_images,)).detach().cpu()

            # Map name to results
            images_dict = {"fake": images}

            # Visualise all results
            for name, images in images_dict.items():
                images_viz = vutils.make_grid(images, padding=2, normalize=True)

                vutils.save_image(
                    images_viz,
                    f"{img_dir}/{name}_samples_step_{global_step}.png",
                    normalize=True,
                )

                # if 'img' not in self.writers:
                #     self.writers['img'] = self._build_writer('img')

                self.writer.add_image(
                    f"{name}_vis",
                    images_viz,
                    global_step=global_step,
                )
                