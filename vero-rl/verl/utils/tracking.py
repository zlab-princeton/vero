# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A unified tracking interface that supports logging data to different backend
"""

import base64
import dataclasses
import html
import json
import numbers
import os
from enum import Enum
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import Any


class Tracking:
    """A unified tracking interface for logging experiment data to multiple backends.

    This class provides a centralized way to log experiment metrics, parameters, and artifacts
    to various tracking backends including WandB, MLflow, SwanLab, TensorBoard, and console.

    Attributes:
        supported_backend: List of supported tracking backends.
        logger: Dictionary of initialized logger instances for each backend.
    """

    supported_backend = [
        "wandb",
        "mlflow",
        "swanlab",
        "vemlp_wandb",
        "tensorboard",
        "console",
        "clearml",
        "trackio",
        "file",
    ]

    def __init__(self, project_name, experiment_name, default_backend: str | list[str] = "console", config=None):
        if isinstance(default_backend, str):
            default_backend = [default_backend]
        for backend in default_backend:
            if backend == "tracking":
                import warnings

                warnings.warn("`tracking` logger is deprecated. use `wandb` instead.", DeprecationWarning, stacklevel=2)
            else:
                assert backend in self.supported_backend, f"{backend} is not supported"

        self.logger = {}

        if "tracking" in default_backend or "wandb" in default_backend:
            import wandb

            # settings = None
            # if config and config["trainer"].get("wandb_proxy", None):
            #     settings = wandb.Settings(https_proxy=config["trainer"]["wandb_proxy"])

            # Force offline mode to avoid network-dependent wandb init failures.
            os.environ["WANDB_MODE"] = "offline"
            settings_kwargs = {"mode": "offline"}
            if config and config["trainer"].get("wandb_proxy", None):
                settings_kwargs["https_proxy"] = config["trainer"]["wandb_proxy"]
            settings = wandb.Settings(**settings_kwargs)
            wandb.init(project=project_name, name=experiment_name, config=config, settings=settings)
            self.logger["wandb"] = wandb

        if "trackio" in default_backend:
            import trackio

            trackio.init(project=project_name, name=experiment_name, config=config)
            self.logger["trackio"] = trackio

        if "mlflow" in default_backend:
            import mlflow

            MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:////tmp/mlruns.db")
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

            # Project_name is actually experiment_name in MLFlow
            # If experiment does not exist, will create a new experiment
            experiment = mlflow.set_experiment(project_name)
            mlflow.start_run(experiment_id=experiment.experiment_id, run_name=experiment_name)
            mlflow.log_params(_compute_mlflow_params_from_objects(config))
            self.logger["mlflow"] = _MlflowLoggingAdapter()

        if "swanlab" in default_backend:
            import swanlab

            SWANLAB_API_KEY = os.environ.get("SWANLAB_API_KEY", None)
            SWANLAB_LOG_DIR = os.environ.get("SWANLAB_LOG_DIR", "swanlog")
            SWANLAB_MODE = os.environ.get("SWANLAB_MODE", "cloud")
            if SWANLAB_API_KEY:
                swanlab.login(SWANLAB_API_KEY)  # NOTE: previous login information will be overwritten

            if config is None:
                config = {}  # make sure config is not None, otherwise **config will raise error
            swanlab.init(
                project=project_name,
                experiment_name=experiment_name,
                config={"FRAMEWORK": "verl", **config},
                logdir=SWANLAB_LOG_DIR,
                mode=SWANLAB_MODE,
            )
            self.logger["swanlab"] = swanlab

        if "vemlp_wandb" in default_backend:
            import volcengine_ml_platform
            from volcengine_ml_platform import wandb as vemlp_wandb

            volcengine_ml_platform.init(
                ak=os.environ["VOLC_ACCESS_KEY_ID"],
                sk=os.environ["VOLC_SECRET_ACCESS_KEY"],
                region=os.environ["MLP_TRACKING_REGION"],
            )

            vemlp_wandb.init(
                project=project_name,
                name=experiment_name,
                config=config,
                sync_tensorboard=True,
            )
            self.logger["vemlp_wandb"] = vemlp_wandb

        if "tensorboard" in default_backend:
            self.logger["tensorboard"] = _TensorboardAdapter(project_name, experiment_name)

        if "console" in default_backend:
            from verl.utils.logger import LocalLogger

            self.console_logger = LocalLogger(print_to_console=True)
            self.logger["console"] = self.console_logger

        if "clearml" in default_backend:
            self.logger["clearml"] = ClearMLLogger(project_name, experiment_name, config)

        if "file" in default_backend:
            self.logger["file"] = FileLogger(project_name, experiment_name)

    def log(self, data, step, backend=None):
        for default_backend, logger_instance in self.logger.items():
            if backend is None or default_backend in backend:
                logger_instance.log(data=data, step=step)

    def __del__(self):
        if "wandb" in self.logger:
            self.logger["wandb"].finish(exit_code=0)
        if "swanlab" in self.logger:
            self.logger["swanlab"].finish()
        if "vemlp_wandb" in self.logger:
            self.logger["vemlp_wandb"].finish(exit_code=0)
        if "tensorboard" in self.logger:
            self.logger["tensorboard"].finish()
        if "clearml" in self.logger:
            self.logger["clearml"].finish()
        if "trackio" in self.logger:
            self.logger["trackio"].finish()
        if "file" in self.logger:
            self.logger["file"].finish()


class ClearMLLogger:
    def __init__(self, project_name: str, experiment_name: str, config):
        self.project_name = project_name
        self.experiment_name = experiment_name

        import clearml

        self._task: clearml.Task = clearml.Task.init(
            task_name=experiment_name,
            project_name=project_name,
            continue_last_task=True,
            output_uri=False,
        )

        self._task.connect_configuration(config, name="Hyperparameters")

    def _get_logger(self):
        return self._task.get_logger()

    def log(self, data, step):
        import numpy as np
        import pandas as pd

        # logs = self._rewrite_logs(data)
        logger = self._get_logger()
        for k, v in data.items():
            title, series = k.split("/", 1)

            if isinstance(v, int | float | np.floating | np.integer):
                logger.report_scalar(
                    title=title,
                    series=series,
                    value=v,
                    iteration=step,
                )
            elif isinstance(v, pd.DataFrame):
                logger.report_table(
                    title=title,
                    series=series,
                    table_plot=v,
                    iteration=step,
                )
            else:
                logger.warning(
                    f'Trainer is attempting to log a value of "{v}" of type {type(v)} for key "{k}". This '
                    f"invocation of ClearML logger's function is incorrect so this attribute was dropped. "
                )

    def finish(self):
        self._task.close()


class FileLogger:
    def __init__(self, project_name: str, experiment_name: str):
        self.project_name = project_name
        self.experiment_name = experiment_name

        self.filepath = os.getenv("VERL_FILE_LOGGER_PATH", None)
        if self.filepath is None:
            root_path = os.path.expanduser(os.getenv("VERL_FILE_LOGGER_ROOT", "."))
            directory = os.path.join(root_path, self.project_name)
            os.makedirs(directory, exist_ok=True)
            self.filepath = os.path.join(directory, f"{self.experiment_name}.jsonl")
            print(f"Creating file logger at {self.filepath}")
        self.fp = open(self.filepath, "w")

    def log(self, data, step):
        data = {"step": step, "data": data}
        self.fp.write(json.dumps(data) + "\n")

    def finish(self):
        self.fp.close()


class _TensorboardAdapter:
    def __init__(self, project_name, experiment_name):
        import os

        from torch.utils.tensorboard import SummaryWriter

        tensorboard_dir = os.environ.get("TENSORBOARD_DIR", f"tensorboard_log/{project_name}/{experiment_name}")
        os.makedirs(tensorboard_dir, exist_ok=True)
        print(f"Saving tensorboard log to {tensorboard_dir}.")
        self.writer = SummaryWriter(tensorboard_dir)

    def log(self, data, step):
        for key in data:
            self.writer.add_scalar(key, data[key], step)

    def finish(self):
        self.writer.close()


class _MlflowLoggingAdapter:
    def log(self, data, step):
        import mlflow

        results = {k.replace("@", "_at_"): v for k, v in data.items()}
        mlflow.log_metrics(metrics=results, step=step)


def _compute_mlflow_params_from_objects(params) -> dict[str, Any]:
    if params is None:
        return {}

    return _flatten_dict(_transform_params_to_json_serializable(params, convert_list_to_dict=True), sep="/")


def _transform_params_to_json_serializable(x, convert_list_to_dict: bool):
    _transform = partial(_transform_params_to_json_serializable, convert_list_to_dict=convert_list_to_dict)

    if dataclasses.is_dataclass(x):
        return _transform(dataclasses.asdict(x))
    if isinstance(x, dict):
        return {k: _transform(v) for k, v in x.items()}
    if isinstance(x, list):
        if convert_list_to_dict:
            return {"list_len": len(x)} | {f"{i}": _transform(v) for i, v in enumerate(x)}
        else:
            return [_transform(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, Enum):
        return x.value

    return x


def _flatten_dict(raw: dict[str, Any], *, sep: str) -> dict[str, Any]:
    import pandas as pd

    ans = pd.json_normalize(raw, sep=sep).to_dict(orient="records")[0]
    assert isinstance(ans, dict)
    return ans


_VALIDATION_HTML_STYLE = """
<style>
  * { box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 0; padding: 16px; color: #202124; background:#f5f7fb; }
  h1 { margin: 0 0 12px 0; font-size: 20px; }
  .meta { color:#5f6368; margin-bottom: 16px; font-size: 13px; }
  .card { border:1px solid #e0e4ec; border-radius:12px; margin: 14px 0; overflow: hidden; box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06); background:#fff; }
  .header { display:flex; justify-content: space-between; align-items:center; padding:10px 14px; background:#f8fafc; border-bottom:1px solid #e8ebf3; flex-wrap:wrap; gap:8px; }
  .pill { display:inline-flex; align-items:center; background:#eef2ff; border-radius:999px; padding:3px 10px; font-size:12px; color:#4338ca; }
  .score-pill { background:#ecfdf3; color:#0f766e; }
  .grid { display:flex; gap:18px; padding:16px; flex-wrap:wrap; }
  .left { flex:1 1 320px; max-width:500px; display:flex; flex-direction:column; gap:8px; }
  .viz { width:100%; border:1px solid #e0e4ec; border-radius:10px; background:#fff; object-fit:contain; max-height:420px; }
  .noimg { width:100%; min-height:160px; display:flex; align-items:center; justify-content:center; border:1px dashed #cdd4e0; border-radius:10px; color:#7b8190; font-size:12px; text-align:center; padding:12px; background:#fafbff; }
  .right { flex:1 1 320px; min-width:260px; }
  .block { margin-bottom: 14px; }
  .title { font-size:12px; color:#6b7280; margin-bottom:6px; text-transform:uppercase; letter-spacing:0.05em; }
  .question { font-size:15px; line-height:1.45; white-space:pre-wrap; }
  pre.box { white-space: pre-wrap; background:#f9fafc; border:1px solid #e3e7f1; border-radius:10px; padding:10px 12px; font-size:13px; line-height:1.4; margin:0; }
  code.path { background:#edf1fb; padding:2px 6px; border-radius:6px; font-size:12px; }
  .footer { font-size:11px; color:#6b7280; margin-top: 4px; }
  .score-val { font-weight:700; color:#0f766e; font-size:14px; }
</style>
"""


def _escape_html_text(value: Any) -> str:
    if value is None:
        return ""
    return html.escape(str(value), quote=False)


def _format_reward_score(score: Any) -> str:
    if score is None:
        return "—"
    if isinstance(score, numbers.Number):
        return f"{float(score):.4f}"
    return _escape_html_text(score)


def _extract_first_image(image: Any) -> Any:
    if isinstance(image, (list, tuple)):
        return image[0] if len(image) > 0 else None
    return image


def _encode_image_bytes(payload: bytes, mime: str) -> str:
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _image_to_data_uri(image: Any) -> tuple[str | None, str | None]:
    """Convert common image formats to a base64 data URI."""
    label = None
    image = _extract_first_image(image)

    if image is None:
        return None, label

    if isinstance(image, (str, os.PathLike, Path)):
        path = Path(image)
        label = str(path)
        if not path.exists():
            return None, label

        suffix = path.suffix.lower()
        mime = "image/png"
        if suffix in {".jpg", ".jpeg"}:
            mime = "image/jpeg"
        elif suffix == ".gif":
            mime = "image/gif"
        elif suffix == ".webp":
            mime = "image/webp"

        try:
            payload = path.read_bytes()
        except OSError:
            return None, label

        return _encode_image_bytes(payload, mime), label

    try:
        from PIL import Image
    except Exception:
        Image = None

    if Image and isinstance(image, Image.Image):
        try:
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            return _encode_image_bytes(buffer.getvalue(), "image/png"), label
        except Exception:
            return None, label

    if isinstance(image, bytes):
        return _encode_image_bytes(image, "image/png"), label

    try:
        import numpy as np  # type: ignore
    except Exception:
        np = None

    if np is not None and isinstance(image, np.ndarray):
        try:
            arr = image
            if arr.ndim == 3 and arr.shape[0] in (1, 3):
                arr = np.transpose(arr, (1, 2, 0))
            arr = np.clip(arr, 0, 255)
            if arr.dtype != np.uint8:
                arr = arr.astype("uint8")
            from PIL import Image as PILImage

            buffer = BytesIO()
            PILImage.fromarray(arr).save(buffer, format="PNG")
            return _encode_image_bytes(buffer.getvalue(), "image/png"), label
        except Exception:
            return None, label

    try:
        import torch  # type: ignore
    except Exception:
        torch = None

    if torch is not None and isinstance(image, torch.Tensor):
        try:
            return _image_to_data_uri(image.detach().cpu().numpy())
        except Exception:
            return None, label

    return None, label


def _render_validation_card(index: int, record: dict[str, Any]) -> str:
    image_block = "<div class='noimg'>No image available</div>"
    image_uri = record.get("image_uri")
    if image_uri:
        image_block = f"<img class='viz' src=\"{image_uri}\" alt=\"sample image\" />"

    footer = ""
    image_label = record.get("image_label")
    if image_label:
        footer = f"<div class='footer'>Image: <code class='path'>{_escape_html_text(image_label)}</code></div>"

    return (
        "<div class='card'>"
        "<div class='header'>"
        f"<span class='pill'>Sample {index + 1}</span>"
        f"<span class='pill score-pill'>Reward: {_format_reward_score(record.get('score'))}</span>"
        "</div>"
        "<div class='grid'>"
        f"<div class='left'>{image_block}{footer}</div>"
        "<div class='right'>"
        "<div class='block'>"
        "<div class='title'>Input</div>"
        f"<div class='question'>{_escape_html_text(record.get('input') or '—')}</div>"
        "</div>"
        "<div class='block'>"
        "<div class='title'>Model Prediction</div>"
        f"<pre class='box'>{_escape_html_text(record.get('output') or '—')}</pre>"
        "</div>"
        "<div class='block'>"
        "<div class='title'>Ground Truth</div>"
        f"<pre class='box'>{_escape_html_text(record.get('ground_truth') or '—')}</pre>"
        "</div>"
        "<div class='block score-block'>"
        "<div class='title'>Reward Score</div>"
        f"<div class='score-val'>{_format_reward_score(record.get('score'))}</div>"
        "</div>"
        "</div>"
        "</div>"
        "</div>"
    )


@dataclasses.dataclass
class ValidationGenerationsLogger:
    project_name: str = None
    experiment_name: str = None

    def log(self, loggers, samples, step, images=None, sample_gts=None):
        if "wandb" in loggers:
            self.log_generations_to_wandb(samples, step, images=images, sample_gts=sample_gts)
        if "swanlab" in loggers:
            self.log_generations_to_swanlab(samples, step)
        if "mlflow" in loggers:
            self.log_generations_to_mlflow(samples, step)

        if "clearml" in loggers:
            self.log_generations_to_clearml(samples, step)
        if "tensorboard" in loggers:
            self.log_generations_to_tensorboard(samples, step)

        if "vemlp_wandb" in loggers:
            self.log_generations_to_vemlp_wandb(samples, step)

    def log_generations_to_vemlp_wandb(self, samples, step):
        from volcengine_ml_platform import wandb as vemlp_wandb

        self._log_generations_to_wandb(samples, step, vemlp_wandb)

    def log_generations_to_wandb(self, samples, step, images=None, sample_gts=None):
        import wandb

        self._log_generations_to_wandb(samples, step, wandb, images=images, sample_gts=sample_gts)
        # self._log_generations_to_wandb_html(samples, step, wandb, images=images, sample_gts=sample_gts)

    def _log_generations_to_wandb(self, samples, step, wandb, images=None, sample_gts=None):
        """Log samples to wandb as a table"""

        print(images)

        # Create column names for all samples
        columns = ["step"] + sum(
            [[f"input_{i + 1}", f"output_{i + 1}", f"score_{i + 1}"] for i in range(len(samples))], []
        )

        if not hasattr(self, "validation_table"):
            # Initialize the table on first call
            self.validation_table = wandb.Table(columns=columns)

        # Create a new table with same columns and existing data
        # Workaround for https://github.com/wandb/wandb/issues/2981#issuecomment-1997445737
        new_table = wandb.Table(columns=columns, data=self.validation_table.data)

        # Add new row with all data
        row_data = []
        row_data.append(step)
        for sample in samples:
            row_data.extend(sample)

        new_table.add_data(*row_data)

        # Update reference and log
        wandb.log({"val/generations": new_table}, step=step)
        self.validation_table = new_table

        print(f"Logged {len(samples)} validation generations to WandB at step {step}.")

    def _log_generations_to_wandb_html(self, samples, step, wandb, images=None, sample_gts=None):
        """Log samples to wandb as an HTML gallery with text, reward, and optional visuals."""
        if not samples:
            return

        records = []
        for idx, sample in enumerate(samples):
            input_text = sample[0] if len(sample) > 0 else ""
            output_text = sample[1] if len(sample) > 1 else ""
            score = sample[2] if len(sample) > 2 else None
            ground_truth = sample_gts[idx] if sample_gts is not None and idx < len(sample_gts) else None
            image_obj = images[idx] if images is not None and idx < len(images) else None
            image_uri, image_label = _image_to_data_uri(image_obj)

            records.append(
                {
                    "input": input_text,
                    "output": output_text,
                    "score": score,
                    "ground_truth": ground_truth,
                    "image_uri": image_uri,
                    "image_label": image_label,
                }
            )

        cards_html = "".join(_render_validation_card(i, record) for i, record in enumerate(records))
        summary = f"Step {step} — showing {len(records)} validation samples."
        gallery_html = (
            "<!DOCTYPE html><html><head><meta charset='utf-8'/>"
            f"{_VALIDATION_HTML_STYLE}"
            "</head><body>"
            "<h1>Validation Generations</h1>"
            f"<div class='meta'>{_escape_html_text(summary)}</div>"
            f"{cards_html}"
            "</body></html>"
        )

        print("Uploading validation generations HTML to WandB...")

        wandb.log({"val/generations_html": wandb.Html(gallery_html)}, step=step)

        print(f"Logged {len(samples)} validation generations as HTML to WandB at step {step}.")

    def log_generations_to_swanlab(self, samples, step):
        """Log samples to swanlab as text"""
        import swanlab

        swanlab_table = swanlab.echarts.Table()

        # Create column names
        headers = ["step", "input", "output", "score"]

        swanlab_row_list = [[step, *sample] for sample in samples]
        swanlab_table.add(headers=headers, rows=swanlab_row_list)

        # Log to swanlab
        swanlab.log({"val/generations": swanlab_table}, step=step)

    def log_generations_to_mlflow(self, samples, step):
        """Log validation generation to mlflow as artifacts"""
        # https://mlflow.org/docs/latest/api_reference/python_api/mlflow.html?highlight=log_artifact#mlflow.log_artifact

        import json
        import tempfile

        import mlflow

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                validation_gen_step_file = Path(tmp_dir, f"val_step{step}.json")
                row_data = []
                for sample in samples:
                    data = {"input": sample[0], "output": sample[1], "score": sample[2]}
                    row_data.append(data)
                with open(validation_gen_step_file, "w") as file:
                    json.dump(row_data, file)
                mlflow.log_artifact(validation_gen_step_file)
        except Exception as e:
            print(f"WARNING: save validation generation file to mlflow failed with error {e}")

    def log_generations_to_clearml(self, samples, step):
        """Log validation generation to clearml as table"""

        import clearml
        import pandas as pd

        task: clearml.Task | None = clearml.Task.current_task()
        if task is None:
            return

        table = [
            {
                "step": step,
                "input": sample[0],
                "output": sample[1],
                "score": sample[2],
            }
            for sample in samples
        ]

        logger = task.get_logger()
        logger.report_table(
            series="Validation generations",
            title="Validation",
            table_plot=pd.DataFrame.from_records(table),
            iteration=step,
        )

    def log_generations_to_tensorboard(self, samples, step):
        """Log samples to tensorboard as text"""
        # Initialize tensorboard writer if not exists
        if not hasattr(self, "writer"):
            from torch.utils.tensorboard import SummaryWriter

            # Use the same directory structure as _TensorboardAdapter
            if self.project_name and self.experiment_name:
                default_dir = os.path.join("tensorboard_log", self.project_name, self.experiment_name)
            else:
                default_dir = "tensorboard_log"

            tensorboard_dir = os.environ.get("TENSORBOARD_DIR", default_dir)
            os.makedirs(tensorboard_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=tensorboard_dir)

        # Format the samples data into readable text
        text_content = f"**Generation Results - Step {step}**\n\n"

        for i, sample in enumerate(samples):
            text_content += f"### Sample {i + 1}\n"

            # Assuming sample contains [input, output, score]
            if len(sample) >= 3:
                input_text, output_text, score = sample[0], sample[1], sample[2]

                text_content += f"**Input:** {input_text}\n\n"
                text_content += f"**Output:** {output_text}\n\n"
                text_content += f"**Score:** {score}\n\n"
            else:
                # Handle cases where sample format might be different
                text_content += f"**Data:** {sample}\n\n"

            text_content += "---\n\n"

        # Log to tensorboard as text
        self.writer.add_text("val/generations", text_content, step)
        # Flush to ensure data is written
        self.writer.flush()
