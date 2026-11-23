// A tiny stack VM for make.fun's tiny stack ISA (see README.md).
import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

// Every instruction's shape: [opcode, immediate bytes, stack pops, stack
// pushes], in opcode order (HALT..EXIT as the switch names them). A byte
// with no row names no instruction (trap 5); pops/pushes drive traps 2/3.
const ROWS: [number, number, number, number][] = [
  [0x00,0,0,0],[0x01,1,0,1],[0x02,4,0,1],[0x03,0,1,0],
  [0x04,0,1,2],[0x05,0,2,2],[0x06,0,2,3],[0x07,0,3,3],
  [0x17,0,1,1],[0x1b,0,1,1],
  [0x30,2,0,0],[0x31,2,1,0],[0x32,2,1,0],
  [0x33,2,0,0],[0x34,0,0,0],
  [0x40,0,1,1],[0x41,0,2,0],
  [0x50,0,1,0],[0x51,0,1,0],[0x52,0,0,1],[0x53,0,1,0],
];
for (const op of [0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x18,0x19,0x1a,0x1c,0x1d,0x1e,0x20,0x21,0x22]) {
  ROWS.push([op,0,2,1]); // the two-in, one-out operations below
}
const SHAPE: [number, number, number][] = [];
for (const r of ROWS) SHAPE[r[0]] = [r[1], r[2], r[3]];

// The two-in, one-out operations: ADD SUB MUL DIV MOD DIVU MODU AND OR XOR
// SHL SHR SHRU EQ LT LTU. The caller checks for a zero divisor first (trap
// 4), so b is never 0 for the divisions; shifts count b's low 5 bits.
const BIN: Record<number, (a: number, b: number) => number> = {
  0x10: (a, b) => (a + b) | 0,
  0x11: (a, b) => (a - b) | 0,
  0x12: (a, b) => Math.imul(a, b), // MUL: the exact low 32 bits
  0x13: (a, b) => (a === -2147483648 && b === -1 ? a : (a / b) | 0), // DIV: toward zero
  0x14: (a, b) => (b === -1 ? 0 : (a % b) | 0), // MOD: takes a's sign
  0x15: (a, b) => Math.floor((a >>> 0) / (b >>> 0)) | 0,
  0x16: (a, b) => ((a >>> 0) % (b >>> 0)) | 0,
  0x18: (a, b) => a & b,
  0x19: (a, b) => a | b,
  0x1a: (a, b) => a ^ b,
  0x1c: (a, b) => a << (b & 31),
  0x1d: (a, b) => a >> (b & 31),
  0x1e: (a, b) => (a >>> (b & 31)) | 0,
  0x20: (a, b) => (a === b ? 1 : 0),
  0x21: (a, b) => (a < b ? 1 : 0),
  0x22: (a, b) => ((a >>> 0) < (b >>> 0) ? 1 : 0),
};

// Runs one program to its end: the bytes it wrote and its exit code (0 for
// HALT, EXIT's low byte, or a trap's code; a trap keeps prior output).
function run(prog: Uint8Array, input: Uint8Array): { bytes: Uint8Array; code: number } {
  const stack = new Int32Array(256);
  const calls = new Int32Array(64);
  const mem = new Int32Array(256);
  const out = new Uint8Array(65536);
  let sp = 0, csp = 0, olen = 0, read = 0, pc = 0, steps = 0;
  // A trap is thrown as its bare code and caught at the end of the run.
  const trap = (code: number): never => { throw code; };
  const finish = (code: number) => ({ bytes: out.slice(0, olen), code });
  try {
    for (;;) {
      if (steps >= 1000000) trap(10);
      if (pc < 0 || pc >= prog.length) trap(9);
      const op = prog[pc];
      const shape = SHAPE[op];
      if (shape === undefined) trap(5);
      if (pc + shape[0] >= prog.length) trap(9);
      steps++;
      if (sp < shape[1]) trap(2);
      if (sp - shape[1] + shape[2] > 256) trap(3);
      const next = pc + 1 + shape[0];
      pc = next;
      if (BIN[op] !== undefined) {
        const b = stack[sp - 1];
        if (b === 0 && op >= 0x13 && op <= 0x16) trap(4);
        stack[sp - 2] = BIN[op](stack[sp - 2], b);
        sp--;
        continue;
      }
      switch (op) {
        case 0x00: return finish(0); // HALT
        case 0x01: stack[sp++] = (prog[next - 1] << 24) >> 24; break; // PUSH8: sign-extended
        case 0x02: // PUSH32: a little-endian signed word
          stack[sp++] = prog[next - 4] | (prog[next - 3] << 8) | (prog[next - 2] << 16) | (prog[next - 1] << 24);
          break;
        case 0x03: sp--; break; // POP
        case 0x04: stack[sp] = stack[sp - 1]; sp++; break; // DUP
        case 0x05: { // SWAP
          const t = stack[sp - 1];
          stack[sp - 1] = stack[sp - 2];
          stack[sp - 2] = t;
          break;
        }
        case 0x06: stack[sp] = stack[sp - 2]; sp++; break; // OVER
        case 0x07: { // ROT: ( a b c -- b c a )
          const a = stack[sp - 3];
          stack[sp - 3] = stack[sp - 2];
          stack[sp - 2] = stack[sp - 1];
          stack[sp - 1] = a;
          break;
        }
        case 0x17: stack[sp - 1] = (-stack[sp - 1]) | 0; break; // NEG: wraps
        case 0x1b: stack[sp - 1] = ~stack[sp - 1]; break; // NOT
        case 0x30: case 0x31: case 0x32: { // JMP, JZ, JNZ: signed relative offset
          const off = ((prog[next - 2] | (prog[next - 1] << 8)) << 16) >> 16;
          // JZ jumps on a popped 0, JNZ on a popped non-0.
          if (op === 0x30 || (stack[--sp] === 0) === (op === 0x31)) pc = next + off;
          break;
        }
        case 0x33: // CALL: the immediate is an absolute address
          if (csp >= 64) trap(7);
          calls[csp++] = next;
          pc = prog[next - 2] | (prog[next - 1] << 8);
          break;
        case 0x34: if (csp === 0) trap(8); pc = calls[--csp]; break; // RET
        case 0x40: case 0x41: { // LOAD / STORE
          const addr = stack[sp - 1];
          if (addr < 0 || addr > 255) trap(6);
          if (op === 0x40) stack[sp - 1] = mem[addr];
          else { mem[addr] = stack[sp - 2]; sp -= 2; }
          break;
        }
        case 0x50: { // PRINT: decimal ASCII, no padding, no newline
          const text = String(stack[--sp]);
          if (olen + text.length > 65536) trap(11);
          for (let i = 0; i < text.length; i++) out[olen++] = text.charCodeAt(i);
          break;
        }
        case 0x51: // PUTC: one byte, the value's low 8 bits
          if (olen >= 65536) trap(11);
          out[olen++] = stack[--sp] & 255;
          break;
        case 0x52: // IN: the next input byte, or -1 once it is used up
          stack[sp++] = read < input.length ? input[read++] : -1;
          break;
        case 0x53: return finish(stack[--sp] & 255); // EXIT
      }
    }
  } catch (e) {
    if (typeof e === 'number') return finish(e);
    throw e;
  }
}

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
for (const id of readdirSync(I).sort()) {
  const prog = readFileSync(join(I, id, 'program.bin'));
  const inPath = join(I, id, 'input.bin'); // may not exist
  const input = existsSync(inPath) ? readFileSync(inPath) : new Uint8Array(0);
  const { bytes, code } = run(prog, input);
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'stdout.bin'), bytes);
  writeFileSync(join(O, id, 'exit.txt'), `${code}\n`);
}
