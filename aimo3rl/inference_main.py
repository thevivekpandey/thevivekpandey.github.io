
import os
import sys
import subprocess
import resource

# Raise soft file descriptor limit to avoid "Too many open files"
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))

os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['HF_TOKEN'] = 'YOUR_HF_TOKEN_HERE'
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TRANSFORMERS_NO_FLAX'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRITON_PTXAS_PATH'] = '/usr/local/cuda/bin/ptxas'
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CC'] = 'gcc'
os.environ['TIKTOKEN_ENCODINGS_BASE'] = '/home/ubuntu/tiktoken'

import gc
import re
import math
import time
import queue
import threading
import contextlib
from typing import Optional
from jupyter_client import KernelManager
from collections import Counter, defaultdict
from concurrent.futures import as_completed, ThreadPoolExecutor

import pandas as pd
import polars as pl

from openai import OpenAI

from openai_harmony import (
    HarmonyEncodingName, 
    load_harmony_encoding, 
    SystemContent, 
    ReasoningEffort, 
    ToolNamespaceConfig, 
    Author, 
    Message, 
    Role, 
    TextContent, 
    Conversation
)

from transformers import set_seed

# Pre-warm Triton cache so subprocess doesn't need to compile
try:
    from triton.backends.nvidia.driver import CudaUtils
    CudaUtils()
except Exception:
    pass

class CFG:

    system_prompt = (
        'You are an elite mathematical problem solver with expertise at the International '
        'Mathematical Olympiad (IMO) level. Your goal is to find the correct answer through '
        'rigorous mathematical reasoning.\n\n'

        '# Problem-Solving Approach:\n'
        '1. UNDERSTAND: Carefully read and rephrase the problem in your own words. '
        'Identify what is given, what needs to be found, and any constraints.\n'
        '2. EXPLORE: Consider multiple solution strategies. Think about relevant theorems, '
        'techniques, patterns, or analogous problems. Don\'t commit to one approach immediately.\n'
        '3. PLAN: Select the most promising approach and outline key steps before executing.\n'
        '4. EXECUTE: Work through your solution methodically. Show all reasoning steps clearly.\n'
        '5. VERIFY: Check your answer by substituting back, testing edge cases, or using '
        'alternative methods. Ensure logical consistency throughout.\n\n'

        '# Mathematical Reasoning Principles:\n'
        '- Break complex problems into smaller, manageable sub-problems\n'
        '- Look for patterns, symmetries, and special cases that provide insight\n'
        '- Use concrete examples to build intuition before generalizing\n'
        '- Consider extreme cases and boundary conditions\n'
        '- If stuck, try working backwards from the desired result\n'
        '- Be willing to restart with a different approach if needed\n\n'

        '# Verification Requirements:\n'
        '- Cross-check arithmetic and algebraic manipulations\n'
        '- Verify that your solution satisfies all problem constraints\n'
        '- Test your answer with simple cases or special values when possible\n'
        '- Ensure dimensional consistency and reasonableness of the result\n\n'

        '# Output Format:\n'
        'The final answer must be a non-negative integer between 0 and 99999.\n'
        'Place your final numerical answer inside \\boxed{}, e.g., \\boxed{42}\n\n'

        'Think step-by-step and show your complete reasoning process. Quality of reasoning '
        'is as important as the final answer.'
    )

    tool_prompt = (
        'Use this tool to execute Python code for:\n'
        '- Complex calculations that would be error-prone by hand\n'
        '- Numerical verification of analytical results\n'
        '- Generating examples or testing conjectures\n'
        '- Visualizing problem structure when helpful\n'
        '- Brute-force verification for small cases\n\n'

        'The environment is a stateful Jupyter notebook. Code persists between executions.\n'
        'Always use print() to display results. Write clear, well-commented code.\n\n'

        'Remember: Code should support your mathematical reasoning, not replace it. '
        'Explain what you\'re computing and why before running code.'
    )

    preference_prompt = (
        'You have access to `math`, `numpy`, and `sympy` for:\n\n'

        '# Symbolic Computation (sympy):\n'
        '- Algebraic manipulation and simplification\n'
        '- Solving equations and systems of equations\n'
        '- Symbolic differentiation and integration\n'
        '- Number theory functions (primes, divisors, modular arithmetic)\n'
        '- Polynomial operations and factorization\n'
        '- Working with mathematical expressions symbolically\n\n'

        '# Numerical Computation (numpy):\n'
        '- Array operations and linear algebra\n'
        '- Efficient numerical calculations for large datasets\n'
        '- Matrix operations and eigenvalue problems\n'
        '- Statistical computations\n\n'

        '# Mathematical Functions (math):\n'
        '- Standard mathematical functions (trig, log, exp)\n'
        '- Constants like pi and e\n'
        '- Basic operations for single values\n\n'

        'Best Practices:\n'
        '- Use sympy for exact symbolic answers when possible\n'
        '- Use numpy for numerical verification and large-scale computation\n'
        '- Combine symbolic and numerical approaches: derive symbolically, verify numerically\n'
        '- Document your computational strategy clearly\n'
        '- Validate computational results against known cases or theoretical bounds'
    )

    served_model_name = 'gpt-oss'
    #model_path = '/kaggle/input/gpt-oss-20b/transformers/default/1'
    #model_path = '/kaggle/input/models/danielhanchen/gpt-oss-20b/transformers/default/1/'
    model_path = 'openai/gpt-oss-120b'

    kv_cache_dtype = 'fp8'
    dtype = 'auto'

    high_problem_timeout = 900
    base_problem_timeout = 300

    notebook_limit = 17400
    server_timeout = 180

    session_timeout = 960
    jupyter_timeout = 60
    sandbox_timeout = 30

    stream_interval = 200
    context_tokens = 16384
    buffer_tokens = 512
    search_tokens = 32
    top_logprobs = 0
    batch_size = 16
    attempts = 3
    workers = 24
    turns = 32
    seed = 42

    gpu_memory_utilization = 0.92
    temperature = 0.5
    min_p = 0.02

def extract_answer(trace: str) -> Optional[int]:
    """Extract the final integer answer from a solution trace.
    Searches for \\boxed{N} first (competition standard), then
    falls back to the last bare integer in the trace.
    """
    # \\boxed{N} — last occurrence wins (model may self-correct)
    boxed = re.findall(r'\\\\boxed\{(\d+)\}', trace)
    if boxed:
        return int(boxed[-1])

    # 'answer is N' / 'answer = N' / 'answer: N'
    m = re.search(
        r'(?:answer|result)\s*(?:is|=|:)\s*(\d+)',
        trace, re.IGNORECASE
    )
    if m:
        return int(m.group(1))

    # Last integer ≤ 6 digits (avoids line-numbers / port numbers)
    ints = re.findall(r'\b(\d{1,6})\b', trace)
    if ints:
        return int(ints[-1])

    return None

class AIMO3Template:

    def __init__(self):

        pass

    def get_system_content(self, system_prompt: str, tool_config: ToolNamespaceConfig) -> SystemContent:

        return (
            SystemContent.new()
            .with_model_identity(system_prompt)
            .with_reasoning_effort(reasoning_effort=ReasoningEffort.HIGH)
            .with_tools(tool_config)
        )

    def apply_chat_template(
        self,
        system_prompt: str,
        user_prompt: str,
        tool_config: ToolNamespaceConfig
    ) -> list[Message]:

        system_content = self.get_system_content(system_prompt, tool_config)
        system_message = Message.from_role_and_content(Role.SYSTEM, system_content)

        user_message = Message.from_role_and_content(Role.USER, user_prompt)

        return [system_message, user_message]

class AIMO3Sandbox:

    def __init__(self, timeout: float):

        self._default_timeout = timeout
        self._owns_kernel = False
        self._client = None
        self._km = None
        
        env = os.environ.copy()
        env['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'
        env['PYDEVD_WARN_EVALUATION_TIMEOUT'] = '0'
        env['JUPYTER_PLATFORM_DIRS'] = '1'
        env['PYTHONWARNINGS'] = 'ignore'
        env['MPLBACKEND'] = 'Agg'
        env['PULP_CBC_CMD'] = 'cbc -silent'

        self._km = KernelManager()
        # Let KernelManager find free ports automatically
        self._km.start_kernel(env=env, extra_arguments=['--Application.log_level=CRITICAL'])

        self._client = self._km.blocking_client()
        self._client.start_channels()
        self._client.wait_for_ready(timeout=self._default_timeout)
        self._owns_kernel = True

        self.execute(
            'import math\n'
            'import numpy\n'
            'import sympy\n'
            'import itertools\n'
            'import collections\n'
            'import mpmath\n'
            'mpmath.mp.dps = 64\n'
            'try:\n'
            '    import pulp; pulp.LpSolverDefault.msg = 0\n'
            'except Exception:\n'
            '    pass\n'
        )

    def _format_error(self, traceback: list[str]) -> str:

        clean_lines = []

        for frame in traceback:
            clean_frame = re.sub(r'\x1b\[[0-9;]*m', '', frame)

            if 'File "' in clean_frame and 'ipython-input' not in clean_frame:
                continue

            clean_lines.append(clean_frame)

        return ''.join(clean_lines)

    def execute(self, code: str, timeout: float | None = None) -> str:

        client = self._client
        effective_timeout = timeout or self._default_timeout
        
        msg_id = client.execute(
            code, 
            store_history=True, 
            allow_stdin=False, 
            stop_on_error=False
        )

        stdout_parts = []
        stderr_parts = []
        
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time

            if elapsed > effective_timeout:
                self._km.interrupt_kernel()

                return f'[ERROR] Execution timed out after {effective_timeout} seconds'

            try:
                msg = client.get_iopub_msg(timeout=1.0)

            except queue.Empty:
                continue

            if msg.get('parent_header', {}).get('msg_id') != msg_id:
                continue

            msg_type = msg.get('msg_type')
            content = msg.get('content', {})

            if msg_type == 'stream':
                text = content.get('text', '')

                if content.get('name') == 'stdout':
                    stdout_parts.append(text)

                else:
                    stderr_parts.append(text)

            elif msg_type == 'error':
                traceback_list = content.get('traceback', [])

                stderr_parts.append(self._format_error(traceback_list))

            elif msg_type in {'execute_result', 'display_data'}:
                data = content.get('data', {})
                text = data.get('text/plain')

                if text:
                    stdout_parts.append(text if text.endswith('\n') else f'{text}\n')

            elif msg_type == 'status':
                if content.get('execution_state') == 'idle':
                    break

        stdout = ''.join(stdout_parts)
        stderr = ''.join(stderr_parts)

        if stderr:
            return f'{stdout.rstrip()}\n{stderr}' if stdout else stderr

        return stdout if stdout.strip() else '[WARN] No output. Use print() to see results.'

    def close(self):

        try:
            if self._client:
                self._client.stop_channels()
        except Exception:
            pass

        if self._owns_kernel and self._km is not None:
            try:
                self._km.shutdown_kernel(now=True)
            except Exception:
                pass

            try:
                self._km.cleanup_resources()
            except Exception:
                pass

    def reset(self):
        
        self.execute(
            '%reset -f\n'
            'import math\n'
            'import numpy\n'
            'import sympy\n'
            'import itertools\n'
            'import collections\n'
            'import mpmath\n'
            'mpmath.mp.dps = 64\n'
            'try:\n'
            '    import pulp; pulp.LpSolverDefault.msg = 0\n'
            'except Exception:\n'
            '    pass\n'

        )

    def __del__(self):

        self.close()

class AIMO3Tool:

    def __init__(self, local_jupyter_timeout: float, tool_prompt: str, sandbox=None):

        self._local_jupyter_timeout = local_jupyter_timeout
        self._tool_prompt = tool_prompt
        self._jupyter_session = sandbox
        
        self._owns_session = sandbox is None
        
        self._execution_lock = threading.Lock()
        self._init_lock = threading.Lock()

    def _ensure_session(self):

        if self._jupyter_session is None:
            with self._init_lock:
                if self._jupyter_session is None:
                    self._jupyter_session = AIMO3Sandbox(timeout=self._local_jupyter_timeout)

    def _ensure_last_print(self, code: str) -> str:

        lines = code.strip().split('\n')

        if not lines:
            return code

        last_line = lines[-1].strip()

        if 'print' in last_line or 'import' in last_line:
            return code

        if not last_line:
            return code

        if last_line.startswith('#'):
            return code

        lines[-1] = 'print(' + last_line + ')'

        return '\n'.join(lines)

    @property
    def instruction(self) -> str:

        return self._tool_prompt

    @property
    def tool_config(self) -> ToolNamespaceConfig:

        return ToolNamespaceConfig(
            name='python', 
            description=self.instruction, 
            tools=[]
        )

    def _make_response(self, output: str, channel: str | None = None) -> Message:

        content = TextContent(text=output)
        author = Author(role=Role.TOOL, name='python')
        message = Message(author=author, content=[content]).with_recipient('assistant')

        if channel:
            message = message.with_channel(channel)

        return message

    def process_sync_plus(self, message: Message) -> list[Message]:

        self._ensure_session()
        raw_script = message.content[0].text
        final_script = self._ensure_last_print(raw_script)

        with self._execution_lock:
            try:
                output = self._jupyter_session.execute(final_script)

            except TimeoutError as exc:
                output = f'[ERROR] {exc}'

        return [self._make_response(output, channel=message.channel)]

class AIMO3Solver:

    def __init__(self, cfg, port: int = 8000):
    
        self.cfg = cfg
        self.port = port
        self.base_url = f'http://0.0.0.0:{port}/v1'
        self.api_key = 'sk-local'
        self.template = AIMO3Template()
        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self.stop_token_ids = self.encoding.stop_tokens_for_assistant_actions()
    
        self._preload_model_weights()
        
        self.server_process = self._start_server()
    
        self.client = OpenAI(
            base_url=self.base_url, 
            api_key=self.api_key, 
            timeout=self.cfg.session_timeout
        )
    
        self._wait_for_server()
        self._initialize_kernels()
    
    def _preload_model_weights(self) -> None:
    
        print(f'Loading model weights from {self.cfg.model_path} into OS Page Cache...')
        start_time = time.time()
        
        files_to_load = []
        total_size = 0
    
        for root, _, files in os.walk(self.cfg.model_path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
    
                if os.path.isfile(file_path):
                    files_to_load.append(file_path)
                    total_size += os.path.getsize(file_path)
    
        def _read_file(path: str) -> None:
    
            with open(path, 'rb') as file_object:
                while file_object.read(1024 * 1024 * 1024):
                    pass
    
        with ThreadPoolExecutor(max_workers=self.cfg.workers) as executor:
            list(executor.map(_read_file, files_to_load))
    
        elapsed = time.time() - start_time
        print(f'Processed {len(files_to_load)} files ({total_size / 1e9:.2f} GB) in {elapsed:.2f} seconds.\n')
    
    def _start_server(self) -> subprocess.Popen:
    
        cmd = [
            sys.executable, 
            '-m', 
            'vllm.entrypoints.openai.api_server', 
            '--seed', 
            str(self.cfg.seed), 
            '--model', 
            self.cfg.model_path, 
            '--served-model-name', 
            self.cfg.served_model_name, 
            '--tensor-parallel-size', 
            '1', 
            '--max-num-seqs', 
            str(self.cfg.batch_size), 
            '--gpu-memory-utilization', 
            str(self.cfg.gpu_memory_utilization), 
            '--host', 
            '0.0.0.0', 
            '--port', 
            str(self.port), 
            '--dtype', 
            self.cfg.dtype, 
            '--kv-cache-dtype', 
            self.cfg.kv_cache_dtype, 
            '--max-model-len', 
            str(self.cfg.context_tokens), 
            '--stream-interval', 
            str(self.cfg.stream_interval), 
            '--async-scheduling',
            '--disable-log-stats',
            '--enable-prefix-caching',
            '--enforce-eager'
        ]
    
        self.log_file = open('vllm_server.log', 'w')
    
        return subprocess.Popen(
            cmd, 
            stdout=self.log_file, 
            stderr=subprocess.STDOUT, 
            start_new_session=True
        )
    
    def _wait_for_server(self):
    
        print('Waiting for vLLM server...')
        start_time = time.time()
    
        for _ in range(self.cfg.server_timeout):
            return_code = self.server_process.poll()
    
            if return_code is not None:
                self.log_file.flush()
    
                with open('vllm_server.log', 'r') as log_file:
                    logs = log_file.read()
    
                raise RuntimeError(f'Server died with code {return_code}. Full logs:\n{logs}\n')
    
            try:
                self.client.models.list()
                elapsed = time.time() - start_time
                print(f'Server is ready (took {elapsed:.2f} seconds).\n')
    
                return
    
            except Exception:
                time.sleep(1)
    
        raise RuntimeError('Server failed to start (timeout).\n')
    
    def _initialize_kernels(self) -> None:
    
        print(f'Initializing {self.cfg.workers} persistent Jupyter kernels...')
        start_time = time.time()
    
        self.sandbox_pool = queue.Queue()
    
        def _create_sandbox(i):
            time.sleep(i * 0.1) # Stagger initialization
            return AIMO3Sandbox(timeout=self.cfg.jupyter_timeout)
    
        with ThreadPoolExecutor(max_workers=self.cfg.workers) as executor:
            futures = [executor.submit(_create_sandbox, i) for i in range(self.cfg.workers)]
    
            for future in as_completed(futures):
                self.sandbox_pool.put(future.result())
    
        elapsed = time.time() - start_time
        print(f'Kernels initialized in {elapsed:.2f} seconds.\n')
    
    def _process_attempt(
        self,
        problem: str,
        system_prompt: str,
        attempt_index: int
    ) -> str:

        local_tool = None
        sandbox = None
        trace_parts = []

        attempt_seed = int(math.pow(self.cfg.seed + attempt_index, 2))

        try:
            sandbox = self.sandbox_pool.get(timeout=self.cfg.sandbox_timeout)

            local_tool = AIMO3Tool(
                local_jupyter_timeout=self.cfg.jupyter_timeout,
                tool_prompt=self.cfg.tool_prompt,
                sandbox=sandbox
            )

            encoding = self.encoding
            messages = self.template.apply_chat_template(
                system_prompt,
                problem,
                local_tool.tool_config
            )

            conversation = Conversation.from_messages(messages)

            for _ in range(self.cfg.turns):
                prompt_ids = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)
                max_tokens = self.cfg.context_tokens - len(prompt_ids)

                if max_tokens < self.cfg.buffer_tokens:
                    break

                stream = self.client.completions.create(
                    model=self.cfg.served_model_name,
                    temperature=self.cfg.temperature,
                    max_tokens=max_tokens,
                    prompt=prompt_ids,
                    seed=attempt_seed,
                    stream=True,
                    extra_body={
                        'min_p': self.cfg.min_p,
                        'stop_token_ids': self.stop_token_ids,
                        'return_token_ids': True
                    }
                )

                try:
                    token_buffer = []
                    text_chunks = []

                    for chunk in stream:
                        new_tokens = chunk.choices[0].token_ids
                        new_text = chunk.choices[0].text

                        if new_tokens:
                            token_buffer.extend(new_tokens)
                            text_chunks.append(new_text)

                finally:
                    stream.close()

                if not token_buffer:
                    break

                assistant_text = ''.join(text_chunks)
                trace_parts.append(f'[ASSISTANT]\n{assistant_text}\n')

                new_messages = encoding.parse_messages_from_completion_tokens(token_buffer, Role.ASSISTANT)
                conversation.messages.extend(new_messages)
                last_message = new_messages[-1]

                if last_message.channel == 'final':
                    break

                if last_message.recipient == 'python':
                    tool_responses = local_tool.process_sync_plus(last_message)
                    response_text = tool_responses[0].content[0].text
                    trace_parts.append(f'[PYTHON OUTPUT]\n{response_text}\n')
                    conversation.messages.extend(tool_responses)

        except Exception as exc:
            trace_parts.append(f'[ERROR]\n{exc}\n')

        finally:
            if sandbox is not None:
                sandbox.reset()
                self.sandbox_pool.put(sandbox)

        return '\n'.join(trace_parts)
    
    def run_attempt(self, problem: str, problem_id: str, attempt_index: int) -> tuple:
        """Run one attempt for a problem.
        Returns (problem_id, attempt_index, answer, trace).
        This is the unit of work handed to the thread pool.
        """
        user_input = f'{problem} {self.cfg.preference_prompt}'
        trace = self._process_attempt(user_input, self.cfg.system_prompt, attempt_index)
        answer = extract_answer(trace)

        output_path = f'./working/{problem_id}-{attempt_index + 1}.txt'
        with open(output_path, 'w') as f:
            f.write(trace)

        #print(f'  [P{problem_id} A{attempt_index + 1}] answer={answer}')
        return problem_id, attempt_index, answer, trace
    
    def __del__(self):
    
        if hasattr(self, 'server_process'):
            self.server_process.terminate()
            self.server_process.wait()
    
        if hasattr(self, 'log_file'):
            self.log_file.close()
    
        if hasattr(self, 'sandbox_pool'):
            while not self.sandbox_pool.empty():
                try:
                    sb = self.sandbox_pool.get_nowait()
                    sb.close()
    
                except Exception:
                    pass

solver = AIMO3Solver(CFG)

def predict(id_: pl.DataFrame, question: pl.DataFrame, answer: Optional[pl.DataFrame] = None) -> pl.DataFrame:

    #id_value = id_.item(0)
    #question_text = question.item(0)

    id_value = id_
    question_text = question
    gc.disable()

    final_answer = solver.solve_problem(question_text, id_value)

    gc.enable()
    gc.collect()

    return pl.DataFrame({'id': id_value, 'answer': final_answer})

import json

NUM_ATTEMPTS = CFG.attempts   # 10
MAX_WORKERS  = CFG.workers    # 32

# ── Load problems ──────────────────────────────────────────────────────────────
problems = []
with open('jee.jsonl', 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        question_text = data['question']
        if 'optionA' in data:
            question_text += (
                f"\n\n(A) {data['optionA']}"
                f"\n(B) {data['optionB']}"
                f"\n(C) {data['optionC']}"
                f"\n(D) {data['optionD']}"
            )
        problems.append((data['id'], question_text))

total_tasks = len(problems) * NUM_ATTEMPTS
print(f'Problems      : {len(problems)}')
print(f'Attempts each : {NUM_ATTEMPTS}')
print(f'Total tasks   : {total_tasks}')
print(f'Workers       : {MAX_WORKERS}')
full_problems = MAX_WORKERS // NUM_ATTEMPTS
partial = MAX_WORKERS % NUM_ATTEMPTS
print(f'Initial fill  : {full_problems} full problems + P{full_problems + 1} gets {partial} slots\n')

# ── Result tracking ────────────────────────────────────────────────────────────
# problem_id -> list of (attempt_index, answer)
all_results   = defaultdict(list)
results_lock  = threading.Lock()

output_dir    = 'working/'
completed_count = 0

def write_problem_summary(problem_id: str, question: str, attempts: list) -> None:
    """Write a summary file once all attempts for a problem are done."""
    attempts_sorted = sorted(attempts, key=lambda x: x[0])   # sort by attempt index
    answers = [a for _, a in attempts_sorted]

    path = f'{output_dir}/{problem_id}_summary.txt'
    with open(path, 'w') as f:
        f.write(f'Problem ID      : {problem_id}\n')
        f.write(f'Question        : {question}\n\n')
        f.write(f'Per-attempt answers:\n')
        for aidx, ans in attempts_sorted:
            f.write(f'  Attempt {aidx + 1:02d}: {ans}\n')
        f.write(f'\nAll answers     : {answers}\n')

    print(f'\n>>> Problem {problem_id} complete | {question[:20]} | answers={answers} <<<\n')

# ── Build work list: all attempts of P1 first, then P2, etc. ──────────────────
# ThreadPoolExecutor queues tasks in submission order, so the first MAX_WORKERS
# tasks fill floor(MAX_WORKERS / NUM_ATTEMPTS) problems fully, plus a partial.
# e.g. 32 workers, 10 attempts → P1(10) + P2(10) + P3(10) + P4(2).
# As any thread finishes, the next queued task starts — naturally greedy.
work_items = [
    (problem_id, question, attempt_index)
    for problem_id, question in problems
    for attempt_index in range(NUM_ATTEMPTS)
]

questions = {pid: q for pid, q in problems}

# ── Execute ────────────────────────────────────────────────────────────────────
solve_start_time = time.time()
last_checkpoint_time = solve_start_time
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    future_map = {
        executor.submit(solver.run_attempt, question, problem_id, attempt_index):
            (problem_id, attempt_index)
        for problem_id, question, attempt_index in work_items
    }

    for future in as_completed(future_map):
        problem_id, attempt_index = future_map[future]
        completed_count += 1
        print(f"one problem done — {time.time() - solve_start_time:.1f}s elapsed")

        try:
            pid, aidx, answer, _ = future.result()

            with results_lock:
                all_results[pid].append((aidx, answer))
                done = len(all_results[pid]) == NUM_ATTEMPTS
                if done:
                    snapshot = list(all_results[pid])
                else:
                    snapshot = None

            if completed_count % 10 == 0:
                now = time.time()
                elapsed = now - solve_start_time
                avg_throughput = completed_count / elapsed
                recent_elapsed = now - last_checkpoint_time
                recent_throughput = 10 / recent_elapsed
                last_checkpoint_time = now
                print(f'  [{completed_count}/{total_tasks}] {elapsed:.1f}s elapsed — avg {avg_throughput:.3f} attempts/s ({1/avg_throughput:.1f}s/attempt) | recent {recent_throughput:.3f} attempts/s ({recent_elapsed/10:.1f}s/attempt)')

            if snapshot is not None:
                write_problem_summary(pid, questions[pid], snapshot)

        except Exception as exc:
            with results_lock:
                all_results[problem_id].append((attempt_index, None))
                done = len(all_results[problem_id]) == NUM_ATTEMPTS
                if done:
                    snapshot = list(all_results[problem_id])

            print(f'[{completed_count}/{total_tasks}] P{problem_id} attempt {attempt_index + 1} FAILED: {exc}')

            if done:
                write_problem_summary(problem_id, questions[problem_id], snapshot)

total_elapsed = time.time() - solve_start_time
print(f'\nAll {total_tasks} tasks done across {len(problems)} problems in {total_elapsed:.1f}s.')
print(f'Final throughput: {completed_count / total_elapsed:.3f} attempts/s ({total_elapsed / max(completed_count, 1):.1f}s/attempt)')

