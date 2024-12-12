#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <cmath>
#include <chrono>
#include <numeric>
#include <iomanip>
#include <random>
#include <algorithm>
#include <string>
#include <cstdint>
#include <cassert>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

// Função para converter __int128 em string
__host__ std::string u128_to_str(unsigned __int128 x) {
    if (x == 0) return "0";
    std::string s;
    while (x > 0) {
        int d = (int)(x % 10);
        s.push_back((char)('0' + d));
        x /= 10;
    }
    std::reverse(s.begin(), s.end());
    return s;
}

// Função auxiliar diff
__host__ __device__ inline uint64_t diff_uint64(uint64_t a, uint64_t b) {
    return (a > b) ? (a - b) : (b - a);
}

// Estrutura para Karmarkar-Karp
struct Element {
    uint64_t value;
    std::vector<std::pair<uint64_t, int>> components;

    Element(uint64_t v) : value(v) {
        components.emplace_back(v, +1);
    }

    Element(uint64_t v, const Element& A, const Element& B) : value(v) {
        for (auto &c : A.components) components.push_back(c);
        for (auto &c : B.components) components.emplace_back(c.first, -c.second);
    }

    bool operator<(const Element& other) const {
        return value < other.value;
    }
};

// Função para ler a instância do arquivo
bool read_instance(const std::string& filename, std::vector<uint64_t>& numbers) {
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "Erro ao abrir o arquivo: " << filename << std::endl;
        return false;
    }

    int64_t N;
    if (!(infile >> N)) {
        std::cerr << "Erro ao ler o número de elementos N." << std::endl;
        return false;
    }

    if (N <= 0) {
        std::cerr << "N <= 0." << std::endl;
        return false;
    }

    numbers.reserve(N);
    for (int64_t i = 0; i < N; ++i) {
        unsigned long long val;
        if (!(infile >> val)) {
            std::cerr << "Erro ao ler o elemento número " << i + 1 << "." << std::endl;
            return false;
        }
        numbers.push_back(static_cast<uint64_t>(val));
    }

    infile.close();
    return true;
}

// Karmarkar-Karp com GRASP na CPU
uint64_t karmarkar_karp_with_grasp(const std::vector<uint64_t>& numbers, std::vector<int>& solution, double alpha, std::mt19937& rng) {
    auto cmp = [](const Element& A, const Element& B) { return A.value < B.value; };
    std::priority_queue<Element, std::vector<Element>, decltype(cmp)> pq(cmp);

    for (const auto& num : numbers) {
        pq.push(Element(num));
    }

    while (pq.size() > 1) {
        Element largest = pq.top();
        pq.pop();

        std::vector<Element> temp;
        while (!pq.empty()) {
            temp.push_back(pq.top());
            pq.pop();
        }

        std::sort(temp.begin(), temp.end(), [](const Element& a, const Element& b) {
            return a.value > b.value;
        });

        uint64_t max_val = temp.front().value;
        uint64_t min_val = temp.back().value;
        uint64_t threshold = max_val - static_cast<uint64_t>(alpha * static_cast<double>(max_val - min_val));

        std::vector<Element> rcl;
        for (const auto& e : temp) {
            if (e.value >= threshold) {
                rcl.push_back(e);
            } else {
                break;
            }
        }

        if (rcl.empty()) {
            // Se RCL estiver vazio, escolhe o segundo maior elemento se possível
            if (temp.size() >= 2) {
                rcl.push_back(temp[1]);
            } else {
                rcl.push_back(temp[0]);
            }
        }

        std::uniform_int_distribution<size_t> dist(0, rcl.size() - 1);
        size_t index = dist(rng);
        Element second = rcl[index];

        // Remove 'second' de temp
        auto it = std::find_if(temp.begin(), temp.end(), [&](const Element& E) {
            return E.value == second.value && E.components.size() == second.components.size();
        });
        if (it != temp.end()) {
            temp.erase(it);
        }

        uint64_t d = (largest.value > second.value) ? (largest.value - second.value) : (second.value - largest.value);
        Element newE(d, largest, second);

        for (const auto& e : temp) {
            pq.push(e);
        }
        pq.push(newE);
    }

    Element finalE = pq.top();
    pq.pop();

    solution.resize(numbers.size(), 0);
    for (const auto& c : finalE.components) {
        uint64_t num = c.first;
        int sign = c.second;
        // Encontra o índice do número original
        auto it = std::find(numbers.begin(), numbers.end(), num);
        if (it != numbers.end()) {
            size_t idx = std::distance(numbers.begin(), it);
            solution[idx] = sign;
        }
    }

    return finalE.value;
}

// Kernel para calcular as somas de todas as soluções na população
__global__ void calc_sums_kernel(const uint64_t* d_numbers, const int* d_solutions,
                                 uint64_t N, uint64_t population_size,
                                 uint64_t* d_sum1, uint64_t* d_sum2) {
    uint64_t s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s < population_size) {
        uint64_t S1 = 0, S2 = 0;
        for (uint64_t i = 0; i < N; i++) {
            int sign = d_solutions[s * N + i];
            uint64_t val = d_numbers[i];
            if (sign == +1) S1 += val;
            else S2 += val;
        }
        d_sum1[s] = S1;
        d_sum2[s] = S2;
    }
}

// Kernel para avaliar movimentos de todas as soluções na população
__global__ void evaluate_moves_kernel(const uint64_t* d_numbers, const int* d_solutions,
                                      uint64_t N, uint64_t population_size,
                                      const uint64_t* d_sum1, const uint64_t* d_sum2,
                                      uint8_t* d_improvements, uint64_t* d_deltas,
                                      curandState* d_states,
                                      int num_alternatives) {
    uint64_t s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s < population_size) {
        uint64_t S1 = d_sum1[s];
        uint64_t S2 = d_sum2[s];
        __int128 old_diff = (S1 > S2) ? (S1 - S2) : (S2 - S1);

        uint64_t best_move_index = 0;
        __int128 best_new_diff = old_diff;
        bool improvement_found = false;

        for (int a = 0; a < num_alternatives; a++) {
            // Gerar um índice aleatório
            unsigned int rand_idx = curand(&d_states[s]);
            uint64_t i = rand_idx % N;

            int sign = d_solutions[s * N + i];
            uint64_t val = d_numbers[i];

            uint64_t new_s1 = (sign == +1) ? (S1 - val) : (S1 + val);
            uint64_t new_s2 = (sign == +1) ? (S2 + val) : (S2 - val);

            __int128 new_diff = (new_s1 > new_s2) ? (new_s1 - new_s2) : (new_s2 - new_s1);

            if (new_diff < best_new_diff) {
                best_new_diff = new_diff;
                best_move_index = i;
                improvement_found = true;
            }
        }

        if (improvement_found) {
            d_improvements[s] = 1;
            d_deltas[s] = best_move_index;
        } else {
            d_improvements[s] = 0;
            d_deltas[s] = 0xFFFFFFFFFFFFFFFFULL;
        }
    }
}

// Kernel para aplicar movimentos melhoradores
__global__ void apply_moves_kernel(int* d_solutions, uint64_t N,
                                   const uint64_t* d_moves, uint64_t moves_count) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < moves_count) {
        uint64_t move = d_moves[idx];
        uint64_t s = move / N;
        uint64_t i = move % N;
        d_solutions[s * N + i] = -d_solutions[s * N + i];
    }
}

// Kernel para inicializar estados do cuRAND
__global__ void init_curand_states_kernel(curandState* states, unsigned long seed, uint64_t total_states) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_states) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

// Função para aplicar melhorias na população
static void multi_improvement_population(const uint64_t* d_numbers,
                                         int* d_solutions,
                                         uint64_t* d_sum1,
                                         uint64_t* d_sum2,
                                         uint8_t* d_improvements,
                                         uint64_t* d_deltas,
                                         curandState* d_states,
                                         uint64_t N,
                                         uint64_t population_size,
                                         uint64_t* d_moves,
                                         uint64_t* d_moves_count,
                                         int top_k,
                                         double subset_ratio,
                                         int num_alternatives,
                                         std::mt19937& rng) {
    // Calcula as somas para todas as soluções
    uint64_t threads_per_block = 256;
    uint64_t blocks = (population_size + threads_per_block - 1) / threads_per_block;
    calc_sums_kernel<<<blocks, threads_per_block>>>(d_numbers, d_solutions, N, population_size, d_sum1, d_sum2);
    cudaDeviceSynchronize();

    // Avalia os movimentos para todas as soluções com múltiplas alternativas
    evaluate_moves_kernel<<<blocks, threads_per_block>>>(d_numbers, d_solutions, N, population_size,
                                                         d_sum1, d_sum2, d_improvements, d_deltas,
                                                         d_states, num_alternatives);
    cudaDeviceSynchronize();

    // Copia os resultados para o host
    std::vector<uint8_t> host_improvements(population_size);
    std::vector<uint64_t> host_deltas(population_size);
    cudaMemcpy(host_improvements.data(), d_improvements, population_size * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_deltas.data(), d_deltas, population_size * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    // Seleciona todos os movimentos melhoradores
    std::vector<uint64_t> moves; // Índices para aplicar
    for (uint64_t s = 0; s < population_size; s++) {
        if (host_improvements[s] == 1) {
            uint64_t i = host_deltas[s];
            moves.push_back(s * N + i); // Representa a posição na matriz de soluções
        }
    }

    if (moves.empty()) return; // Sem melhorias

    // Ordena os movimentos pelo índice (opcional, pode ser removido se não for necessário)
    std::sort(moves.begin(), moves.end());

    // Limita para top_k movimentos
    if (static_cast<int>(moves.size()) > top_k) {
        moves.resize(top_k);
    }

    // Determina quantos movimentos aplicar
    size_t num_to_apply = static_cast<size_t>(moves.size() * subset_ratio);
    if (num_to_apply == 0 && !moves.empty()) num_to_apply = 1;

    // Seleciona aleatoriamente os movimentos a aplicar
    std::shuffle(moves.begin(), moves.end(), rng);
    moves.resize(num_to_apply);

    // Copia os movimentos para o device
    if (num_to_apply > 0) {
        cudaMemcpy(d_moves, moves.data(), num_to_apply * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_moves_count, &num_to_apply, sizeof(uint64_t), cudaMemcpyHostToDevice);

        // Aplica os movimentos na GPU
        uint64_t apply_blocks = (num_to_apply + threads_per_block - 1) / threads_per_block;
        apply_moves_kernel<<<apply_blocks, threads_per_block>>>(d_solutions, N, d_moves, num_to_apply);
        cudaDeviceSynchronize();
    }
}

int main(int argc, char* argv[]) {
    if (argc < 9) {
        std::cerr << "Uso: " << argv[0] << " <instancia> <alpha> <iteracoes> <top_k> <subset_ratio> <max_no_improve> <seed> <num_alternatives>" << std::endl;
        return 1;
    }

    std::string filename = argv[1];
    double alpha = std::stod(argv[2]);            // Parâmetro GRASP
    int iterations = std::stoi(argv[3]);          // Número máximo de iterações
    int top_k = std::stoi(argv[4]);               // Quantos movimentos considerar
    double subset_ratio = std::stod(argv[5]);     // Proporção dos movimentos a aplicar
    int max_no_improve = std::stoi(argv[6]);      // Máximo de iterações sem melhoria
    unsigned int seed = std::stoi(argv[7]);       // Seed para RNG
    int num_alternatives = std::stoi(argv[8]);    // Número de alternativas a considerar por solução

    std::vector<uint64_t> numbers;
    if (!read_instance(filename, numbers)) return 1;

    // Inicializa o gerador de números aleatórios
    std::mt19937 rng(seed);

    // Ordena os números em ordem decrescente
    std::sort(numbers.begin(), numbers.end(), std::greater<uint64_t>());

    // Define o tamanho da população
    uint64_t population_size = 1024; // Pode ser ajustado conforme necessário

    // Gera a população inicial usando Karmarkar-Karp com GRASP na CPU
    std::vector<int> host_solutions(population_size * numbers.size(), +1);
    for (uint64_t s = 0; s < population_size; s++) {
        std::vector<int> solution;
        karmarkar_karp_with_grasp(numbers, solution, alpha, rng);
        for (size_t i = 0; i < solution.size(); i++) {
            host_solutions[s * numbers.size() + i] = solution[i];
        }
    }

    uint64_t N = numbers.size();

    // Aloca e copia os dados para a GPU
    uint64_t* d_numbers;
    cudaError_t err = cudaMalloc(&d_numbers, N * sizeof(uint64_t));
    if (err != cudaSuccess) {
        std::cerr << "Erro ao alocar d_numbers: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    cudaMemcpy(d_numbers, numbers.data(), N * sizeof(uint64_t), cudaMemcpyHostToDevice);

    int* d_solutions;
    err = cudaMalloc(&d_solutions, population_size * N * sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "Erro ao alocar d_solutions: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    cudaMemcpy(d_solutions, host_solutions.data(), population_size * N * sizeof(int), cudaMemcpyHostToDevice);

    uint64_t* d_sum1;
    cudaMalloc(&d_sum1, population_size * sizeof(uint64_t));
    uint64_t* d_sum2;
    cudaMalloc(&d_sum2, population_size * sizeof(uint64_t));
    uint8_t* d_improvements;
    cudaMalloc(&d_improvements, population_size * sizeof(uint8_t));
    uint64_t* d_deltas;
    cudaMalloc(&d_deltas, population_size * sizeof(uint64_t));
    uint64_t* d_moves;
    cudaMalloc(&d_moves, population_size * sizeof(uint64_t)); // Máximo possível
    uint64_t* d_moves_count;
    cudaMalloc(&d_moves_count, sizeof(uint64_t));

    // Inicializa os estados do cuRAND
    curandState* d_states;
    uint64_t total_states = population_size;
    uint64_t threads_per_block = 256;
    uint64_t blocks = (total_states + threads_per_block - 1) / threads_per_block;
    cudaMalloc(&d_states, total_states * sizeof(curandState));
    init_curand_states_kernel<<<blocks, threads_per_block>>>(d_states, seed, total_states);
    cudaDeviceSynchronize();

    auto start = std::chrono::high_resolution_clock::now();

    uint64_t best_diff = UINT64_MAX;
    std::vector<int> best_solution(N, +1);
    int no_improve = 0;

    for (int iter = 0; iter < iterations; iter++) {
        // Aplica Multi-Improvement na população
        multi_improvement_population(d_numbers, d_solutions, d_sum1, d_sum2, d_improvements, d_deltas,
                                     d_states, N, population_size, d_moves, d_moves_count,
                                     top_k, subset_ratio, num_alternatives, rng);

        // Recalcula as somas após as melhorias
        calc_sums_kernel<<<blocks, threads_per_block>>>(d_numbers, d_solutions, N, population_size, d_sum1, d_sum2);
        cudaDeviceSynchronize();

        // Copia as somas para o host
        std::vector<uint64_t> host_sum1(population_size);
        std::vector<uint64_t> host_sum2(population_size);
        cudaMemcpy(host_sum1.data(), d_sum1, population_size * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(host_sum2.data(), d_sum2, population_size * sizeof(uint64_t), cudaMemcpyDeviceToHost);

        // Encontra a melhor solução na população atual
        uint64_t local_best = UINT64_MAX;
        uint64_t best_idx = 0;
        for (uint64_t s = 0; s < population_size; s++) {
            __int128 diff = (host_sum1[s] > host_sum2[s]) ? (host_sum1[s] - host_sum2[s]) : (host_sum2[s] - host_sum1[s]);
            uint64_t abs_diff = static_cast<uint64_t>(diff);
            if (abs_diff < local_best) {
                local_best = abs_diff;
                best_idx = s;
            }
        }

        // Atualiza a melhor solução global
        if (local_best < best_diff) {
            best_diff = local_best;
            cudaMemcpy(best_solution.data(), d_solutions + best_idx * N, N * sizeof(int), cudaMemcpyDeviceToHost);
            no_improve = 0;
        } else {
            no_improve++;
        }

        // Verifica condição de parada
        if (no_improve > max_no_improve) break;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    // Calcula as somas finais
    __int128 S1 = 0, S2 = 0;
    for (size_t i = 0; i < N; i++) {
        if (best_solution[i] == +1) {
            S1 += numbers[i];
        } else {
            S2 += numbers[i];
        }
    }
    __int128 final_diff = (S1 > S2) ? (S1 - S2) : (S2 - S1);

    // Monta os subconjuntos
    std::vector<uint64_t> subset1, subset2;
    for (size_t i = 0; i < N; i++) {
        if (best_solution[i] == +1) {
            subset1.push_back(numbers[i]);
        } else {
            subset2.push_back(numbers[i]);
        }
    }

    // Exibe os resultados
    std::cout << "Diferença mínima possível: " << u128_to_str(static_cast<unsigned __int128>(final_diff)) << "\n";
    std::cout << "Subconjunto 1: ";
    for (auto v : subset1) std::cout << v << " ";
    std::cout << "\nSubconjunto 2: ";
    for (auto v : subset2) std::cout << v << " ";
    std::cout << "\n";
    std::cout << "Soma Subconjunto 1: " << u128_to_str(static_cast<unsigned __int128>(S1)) << "\n";
    std::cout << "Soma Subconjunto 2: " << u128_to_str(static_cast<unsigned __int128>(S2)) << "\n";
    std::cout << "Diferença real: " << u128_to_str(static_cast<unsigned __int128>(final_diff)) << "\n";
    std::cout << "Tempo de execução: " << std::fixed << std::setprecision(6) << duration.count() << " s\n";

    // Libera a memória alocada na GPU
    cudaFree(d_numbers);
    cudaFree(d_solutions);
    cudaFree(d_sum1);
    cudaFree(d_sum2);
    cudaFree(d_improvements);
    cudaFree(d_deltas);
    cudaFree(d_moves);
    cudaFree(d_moves_count);
    cudaFree(d_states);

    return 0;
}
