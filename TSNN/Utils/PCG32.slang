/*
 * The PCG random number generator was developed by Melissa O'Neill
 * <oneill@pcg-random.org>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * For additional information about the PCG random number generation scheme,
 * including its license and other licensing options, visit
 *
 *     http://www.pcg-random.org
 */

struct PCG32
{
    uint64_t state, inc;

    __init(uint64_t initState, uint64_t initSeq)
    {
        state = 0;
        inc   = (initSeq << 1u) | 1u;
        nextUInt();
        state += initState;
        nextUInt();
    }

    [mutating]
    uint nextUInt()
    {
        uint64_t old = state;
        state = old * 0x5851f42d4c957f2dULL + inc;
        uint xorshifted = (uint)(((old >> 18u) ^ old) >> 27u);
        uint rot        = (uint)(old >> 59u);
        return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31));
    }

    [mutating]
    float nextFloat()
    {
        uint x = (nextUInt() >> 9) | 0x3f800000u;
        return reinterpret<float>(x) - 1.0f;
    }
}
