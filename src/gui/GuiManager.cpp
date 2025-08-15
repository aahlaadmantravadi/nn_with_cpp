// =============================================================================
// File: src/gui/GuiManager.cpp (Final Fixed Version)
// =============================================================================
//
// Description: Implements the GuiManager class. This file contains all the
//              logic for window creation, setting up ImGui, rendering the UI
//              components, and handling the main application loop, including
//              parsing commands and running the training process.
//
// =============================================================================

#include "gui/GuiManager.h"



#include "data/DataManager.h"
#include "gui/Visualizer.h"
#include "nn/Loss.h"
#include "nn/Model.h"
#include "nn/layers/Activation.h"
#include "nn/layers/Dense.h"
#include "nn/layers/Softmax.h"
#include "nn/optimizers/Adam.h"
#include "nn/optimizers/SGD.h"
#include "nlp/Parser.h"



#include <GLFW/glfw3.h>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"



#include <atomic>
#include <chrono>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>



#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif



// --- Global state for training thread ---
std::atomic<bool> isTraining(false);
std::thread trainingThread;



// GLFW error callback function
static void glfw_error_callback(int error, const char *description)
{
    std::cerr << "GLFW Error " << error << ": " << description << std::endl;
}



// Helper function to render selectable, multi-line wrapped text
void RenderSelectableWrappedText(const char *label, const std::string &text)
{
    ImGui::PushStyleColor(ImGuiCol_FrameBg, IM_COL32(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_Border, IM_COL32(0, 0, 0, 0));
    ImGui::InputTextMultiline(label, (char *)text.c_str(), text.length() + 1, ImVec2(-1.0f, ImGui::GetTextLineHeight() * 4), ImGuiInputTextFlags_ReadOnly);
    ImGui::PopStyleColor(2);
}



// --- GuiManager Implementation ---

GuiManager::GuiManager()
    : window{nullptr}, windowWidth{1600}, windowHeight{900},
      selectedBackend{Backend::CPU}, showVisualizerWindow{true}, uiScale{2.0f} // Default scale to 2.0x
{
    memset(nlpInputBuffer, 0, sizeof(nlpInputBuffer));
    memset(assistantInputBuffer, 0, sizeof(assistantInputBuffer));
    addLog("Welcome to TensorFlow from Scratch!");
}



GuiManager::~GuiManager()
{
    isTraining = false;
    if (trainingThread.joinable())
    {
        trainingThread.join();
    }
}



void GuiManager::run()
{
    try
    {
        init();
        mainLoop();
        shutdown();
    }
    catch (const std::exception &e)
    {
        throw std::runtime_error(std::string("Failed in GuiManager::run(): ") + e.what());
    }
}



void GuiManager::init()
{
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit()) throw std::runtime_error("Failed to initialize GLFW");

    const char *glsl_version = "#version 330";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    window = glfwCreateWindow(windowWidth, windowHeight, "TensorFlow from Scratch", nullptr, nullptr);
    if (!window)
    {
        glfwTerminate();
        throw std::runtime_error("Failed to create GLFW window");
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    model = std::make_unique<Model>();
    dataManager = std::make_unique<DataManager>();
    nlpParser = std::make_unique<Parser>();
    visualizer = std::make_unique<Visualizer>();

    // Detect CUDA availability
    int device_count = 0;
    cudaError_t cuda_status = cudaGetDeviceCount(&device_count);
    hasCuda = (cuda_status == cudaSuccess && (device_count > 0));
    if (hasCuda)
    {
        selectedBackend = Backend::GPU;
        addLog("CUDA device detected. Defaulting backend to GPU.");
    }
    else
    {
        addLog("No CUDA device detected. Using CPU backend.");
    }

    addLog("GUI Manager initialized. Enter a command to begin.");
}



void GuiManager::shutdown()
{
    isTraining = false;
    if (trainingThread.joinable())
    {
        trainingThread.join();
    }
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    std::cout << "Application shut down gracefully." << std::endl;
}



void GuiManager::mainLoop()
{
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        renderUI();
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.1f, 0.1f, 0.12f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }
}



void GuiManager::renderUI()
{
    ImGui::GetIO().FontGlobalScale = uiScale;

    renderMenuBar();
    renderControlPanel();
    renderLogPanel();
    if (showVisualizerWindow)
    {
        renderVisualizationWindow();
    }
}



void GuiManager::renderMenuBar()
{
    if (ImGui::BeginMainMenuBar())
    {
        if (ImGui::BeginMenu("File"))
        {
            if (ImGui::MenuItem("Exit")) { glfwSetWindowShouldClose(window, true); }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("View"))
        {
            ImGui::MenuItem("Visualizer", NULL, &showVisualizerWindow);
            ImGui::Separator();
            ImGui::Text("UI Scale");
            ImGui::SliderFloat("##ui_scale", &uiScale, 0.5f, 2.5f);
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }
}



void GuiManager::renderControlPanel()
{
    ImGui::Begin("Control Panel");
    ImGui::Text("Natural Language Network Configuration");
    ImGui::Separator();

    if (ImGui::InputText("Command", nlpInputBuffer, sizeof(nlpInputBuffer), ImGuiInputTextFlags_EnterReturnsTrue))
    {
        processNlpInput();
    }
    ImGui::SameLine();
    if (ImGui::Button("Parse & Build"))
    {
        processNlpInput();
    }
    RenderSelectableWrappedText("##eg1", "e.g., 'build 784-128-relu-64-relu-10-softmax with adam for mnist'");
    RenderSelectableWrappedText("##eg2", "e.g., 'train a network to classify handwritten digits'");

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Text("Assistant Chat");
    ImGui::PushItemWidth(-1);
    bool sent = ImGui::InputText("##assistant_input", assistantInputBuffer, sizeof(assistantInputBuffer), ImGuiInputTextFlags_EnterReturnsTrue);
    ImGui::PopItemWidth();
    ImGui::SameLine();
    if (ImGui::Button("Send"))
    {
        sent = true;
    }
    if (sent && (strlen(assistantInputBuffer) > 0))
    {
        std::string user_msg(assistantInputBuffer);
        addLog(std::string("[You] ") + user_msg);
        // Emit to stdout with a clear prefix so the assistant can read it from terminal
        std::cout << "[ASSISTANT_INPUT] " << user_msg << std::endl;
        memset(assistantInputBuffer, 0, sizeof(assistantInputBuffer));
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Text("Training Controls");
    ImGui::Text("CUDA: %s", hasCuda ? "Available" : "Not available");

    if (ImGui::RadioButton("CPU", selectedBackend == Backend::CPU)) { selectedBackend = Backend::CPU; }
    ImGui::SameLine();
    if (ImGui::RadioButton("GPU (CUDA)", selectedBackend == Backend::GPU)) { selectedBackend = Backend::GPU; }
    if (!hasCuda)
    {
        ImGui::SameLine();
        ImGui::TextDisabled("(no device)");
    }

    ImGui::Spacing();

    // Add epoch input control
    ImGui::Text("Epochs:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(60);
    ImGui::InputInt("##epochs", &numEpochs, 1, 5);
    ImGui::SameLine();
    ImGui::Text("[%d]", numEpochs);

    // Batch size control
    ImGui::Text("Batch size:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(120);
    int bs_i = static_cast<int>(batchSize);
    if (ImGui::InputInt("##batchsize", &bs_i, 16, 64))
    {
        if (bs_i < 1) bs_i = 1;
        if (bs_i > 1024) bs_i = 1024;
        batchSize = static_cast<size_t>(bs_i);
    }
    // Learning rate control (wider as requested)
    ImGui::Text("LR:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(800);
    ImGui::InputFloat("##lr", &learningRate, 0.0005f, 0.001f, "%.6f");

    // Debug toggle
    ImGui::SameLine();
    ImGui::Checkbox("Debug", &debugVerbose);

    // Clamp to reasonable values
    if (numEpochs < 1) numEpochs = 1;
    if (numEpochs > 100) numEpochs = 100;

    ImGui::Spacing();

    // Make buttons wider as requested by user
    float buttonWidth = 240;

    if (ImGui::Button("Start Training", ImVec2(buttonWidth, 30)))
    {
        if (!isTraining)
        {
            // Reset training metrics for new session
            currentLoss = 0.0f;
            showTestResults = false;
            currentEpoch = 0;
            currentBatchIndex = 0;
            isTraining = true;

            // Set the selected backend before starting training
            try
            {
                if (selectedBackend == Backend::GPU)
                {
                    addLog("Using GPU backend for training...");
                    model->setBackend(Backend::GPU);
                }
                else
                {
                    addLog("Using CPU backend for training...");
                    model->setBackend(Backend::CPU);
                }

                // Create stable shared copies of data for the thread
                auto *model_ptr = model.get();
                auto *data_ptr = dataManager.get();
                auto *log_ptr = &logMessages;
                auto epochs_to_use = numEpochs; // Capture epochs from UI
                auto *current_loss_ptr = &currentLoss;
                auto *current_epoch_ptr = &currentEpoch;
                auto *current_batch_index_ptr = &currentBatchIndex;
                auto *num_batches_per_epoch_ptr = &numBatchesPerEpoch;
                auto *batch_size_ptr = &batchSize;

                // Detach any previous thread
                if (trainingThread.joinable())
                {
                    trainingThread.detach();
                }

                // Create new thread with captured pointers rather than 'this'
                bool dbg = debugVerbose;
                trainingThread = std::thread([model_ptr, data_ptr, log_ptr, epochs_to_use, current_loss_ptr, current_epoch_ptr, current_batch_index_ptr, num_batches_per_epoch_ptr, batch_size_ptr, dbg]()
                {
                    try
                    {
                        log_ptr->push_back("Training started...");
                        std::cout << "[APP_LOG] Training started..." << std::endl;
                        size_t batch_size = *batch_size_ptr;
                        size_t num_batches = data_ptr->getTrainSamplesCount() / batch_size;
                        *num_batches_per_epoch_ptr = num_batches;

                        for (size_t i = 0; i < epochs_to_use && isTraining; i++)
                        {
                            *current_epoch_ptr = static_cast<int>(i) + 1;
                            float epoch_loss = 0;
                            for (size_t j = 0; j < num_batches && isTraining; j++)
                            {
                                *current_batch_index_ptr = j + 1;
                                try
                                {
                                    auto batch = data_ptr->getTrainBatch(batch_size);
                                    if (dbg)
                                    {
                                        std::cout << "[APP_LOG][DBG] Batch " << (j + 1) << "/" << num_batches
                                                  << ", X:(" << batch.first.getRows() << "," << batch.first.getCols() << ")"
                                                  << ", y:(" << batch.second.getRows() << "," << batch.second.getCols() << ")" << std::endl;
                                    }
                                    epoch_loss += model_ptr->train_step(batch.first, batch.second);
                                }
                                catch (const std::exception &e)
                                {
                                    log_ptr->push_back("Error in training batch: " + std::string(e.what()));
                                    std::cout << "[APP_LOG] Error in training batch: " << e.what() << std::endl;
                                    // Continue with next batch
                                }
                            }

                            float avg_loss = epoch_loss / num_batches;
                            *current_loss_ptr = avg_loss; // Update the loss for UI
                            std::string epoch_msg = "Epoch " + std::to_string(i + 1) + " Loss: " + std::to_string(avg_loss);
                            log_ptr->push_back(epoch_msg);
                            std::cout << "[APP_LOG] " << epoch_msg << std::endl;
                        }

                        isTraining = false;
                        log_ptr->push_back("Training finished.");
                        std::cout << "[APP_LOG] Training finished." << std::endl;
                    }
                    catch (const std::exception &e)
                    {
                        log_ptr->push_back("Training error: " + std::string(e.what()));
                        std::cout << "[APP_LOG] Training error: " << e.what() << std::endl;
                        isTraining = false;
                    }
                });

                // Detach the thread to prevent crashes
                trainingThread.detach();
            }
            catch (const std::exception &e)
            {
                addLog("Error starting training: " + std::string(e.what()));
                isTraining = false;
            }
        }
    }

    ImGui::Spacing();

    if (ImGui::Button("Stop Training", ImVec2(buttonWidth, 30)))
    {
        if (isTraining)
        {
            addLog("Stopping training...");
            isTraining = false;

            // Wait for training thread to finish with a timeout
            auto join_result = std::future<void>();
            if (trainingThread.joinable())
            {
                // Use a separate thread to join with a timeout to avoid deadlock
                std::thread([&trainingThread = trainingThread]()
                {
                    if (trainingThread.joinable())
                    {
                        trainingThread.join();
                    }
                }).detach();
            }

            addLog("Training stopped.");
        }
    }

    ImGui::Spacing();

    if (ImGui::Button("Test Model", ImVec2(buttonWidth, 30)))
    {
        if (model && dataManager && (!isTraining))
        {
            try
            {
                addLog("Testing model on test data...");

                // Get the test data
                auto X_test = dataManager->getTestData();
                auto y_test = dataManager->getTestLabels();

                if (debugVerbose)
                {
                    std::cout << "[APP_LOG][DBG] Test shapes X:(" << X_test.getRows() << "," << X_test.getCols()
                              << ") y:(" << y_test.getRows() << "," << y_test.getCols() << ")" << std::endl;
                }

                auto t0 = std::chrono::steady_clock::now();
                // Evaluate the model
                auto [loss, accuracy] = model->evaluate(X_test, y_test);
                auto t1 = std::chrono::steady_clock::now();
                auto ms_total = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

                // Always log tested sample count; add perf when debug is on
                addLog("Evaluated " + std::to_string(X_test.getRows()) + " samples.");
                if (debugVerbose)
                {
                    double samples_per_sec = (ms_total > 0) ? (static_cast<double>(X_test.getRows()) * 1000.0 / ms_total) : 0.0;
                    std::cout << "[APP_LOG][DBG] Eval total ms:" << ms_total << ", samples/s:" << samples_per_sec << std::endl;
                }

                // Update metrics
                testLoss = loss;
                testAccuracy = accuracy;
                showTestResults = true;

                addLog("Test Loss: " + std::to_string(testLoss));
                addLog("Test Accuracy: " + std::to_string(testAccuracy * 100.0f) + "%");
            }
            catch (const std::exception &e)
            {
                addLog("Error testing model: " + std::string(e.what()));
            }
        }
        else if (isTraining)
        {
            addLog("Cannot test while training is in progress. Stop training first.");
        }
        else
        {
            addLog("No model or data available for testing.");
        }
    }

    ImGui::Spacing();

    ImGui::Separator();
    ImGui::Text("Metrics");

    // Show current training loss if available
    if (currentLoss > 0.0f)
    {
        ImGui::Text("Current Training Loss: %.6f", currentLoss);
    }

    // Show test results if available
    if (showTestResults)
    {
        ImGui::Text("Test Loss: %.6f", testLoss);
        ImGui::Text("Test Accuracy: %.2f%%", testAccuracy * 100.0f);
    }

    if (isTraining)
    {
        ImGui::Spacing();
        ImGui::Text("Progress: Epoch %d/%d, Batch %zu/%zu, BatchSize %zu", currentEpoch, numEpochs, currentBatchIndex, numBatchesPerEpoch, batchSize);
    }

    // Add some space
    ImGui::Dummy(ImVec2(0, 15));

    // Add bottom drag handle
    renderDragHandle("ctrl");
    ImGui::End();
}



void GuiManager::renderLogPanel()
{
    ImGui::Begin("Log");

    // Main content
    for (size_t i = 0; i < logMessages.size(); i++)
    {
        std::string label = "##log" + std::to_string(i);
        RenderSelectableWrappedText(label.c_str(), logMessages[i]);
    }
    if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY())
    {
        ImGui::SetScrollHereY(1.0f);
    }

    // Add some padding
    ImGui::Dummy(ImVec2(0, 10));

    // Add bottom drag handle
    renderDragHandle("log");
    ImGui::End();
}



void GuiManager::renderVisualizationWindow()
{
    ImGui::Begin("Network Visualizer", &showVisualizerWindow);
    ImGui::Text("Live Training Visualization");
    ImGui::Separator();
    if (visualizer)
    {
        visualizer->render(model.get());
    }

    // Add some padding
    ImGui::Dummy(ImVec2(0, 10));

    // Add bottom drag handle
    renderDragHandle("vis");
    ImGui::End();
}



void GuiManager::renderDragHandle(const char *id)
{
    // Position the drag handle at the bottom of the window, using window coordinates
    ImVec2 window_pos = ImGui::GetWindowPos();
    ImVec2 window_size = ImGui::GetWindowSize();

    // Create a full-width button at the bottom of the window
    float height = 10.0f * uiScale;

    // Use SetCursorPos to ensure we position at bottom regardless of scroll position
    ImGui::SetCursorPosY(ImGui::GetWindowHeight() - height - ImGui::GetStyle().WindowPadding.y);

    // Add style for the drag handle
    ImGui::PushStyleColor(ImGuiCol_Button, IM_COL32(60, 60, 70, 255));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, IM_COL32(70, 70, 80, 255));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, IM_COL32(80, 80, 90, 255));
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 0.0f);

    // Create a unique ID for each drag handle
    std::string handle_id = "##draghandle_" + std::string(id);

    // Get the button width - ensure it fills the width regardless of content
    float width = ImGui::GetContentRegionAvail().x;

    // Draw the button
    bool clicked = ImGui::Button(handle_id.c_str(), ImVec2(width, height));

    // Get the button position for drawing grip dots
    ImVec2 button_pos = ImGui::GetItemRectMin();
    ImVec2 button_size = ImGui::GetItemRectSize();

    // Draw grip dots
    ImDrawList *draw_list = ImGui::GetWindowDrawList();
    float dot_spacing = 8.0f * uiScale;

    // Center the dots horizontally
    float start_x = button_pos.x + (button_size.x / 2.0f) - (2.0f * dot_spacing);
    float y = button_pos.y + (button_size.y / 2.0f);

    for (int i = 0; i < 5; i++)
    {
        draw_list->AddCircleFilled(
            ImVec2(start_x + (i * dot_spacing), y),
            1.5f * uiScale,
            IM_COL32(200, 200, 200, 180)
        );
    }

    ImGui::PopStyleVar();

    // Make the entire window draggable when interacting with the handle
    if (ImGui::IsItemActive() && ImGui::IsMouseDragging(0))
    {
        ImGui::SetWindowFocus();
        ImVec2 windowPos = ImGui::GetWindowPos();
        ImVec2 delta = ImGui::GetIO().MouseDelta;
        ImGui::SetWindowPos(ImVec2(windowPos.x + delta.x, windowPos.y + delta.y));
    }

    ImGui::PopStyleColor(3);
}



void GuiManager::processNlpInput()
{
    if (strlen(nlpInputBuffer) == 0) return;

    std::string command(nlpInputBuffer);
    addLog("AI-parsing command: " + command);

    ModelConfig config = nlpParser->parse(command);

    if (config.valid)
    {
        // Step 1: Load dataset using AI-resolved dataset info
        addLog("Loading AI-resolved dataset: " + config.dataset_info.name);

        bool dataset_loaded = false;
        if ((!config.dataset_info.name.empty()) && (config.dataset_info.name != "custom_needed"))
        {
            // Use new AI-driven dataset loading
            dataset_loaded = dataManager->loadDatasetFromInfo(config.dataset_info);
        }
        else
        {
            // Fallback to legacy datasets only if explicitly requested
            if (config.dataset == "mnist")
            {
                dataset_loaded = dataManager->loadDataset(Dataset::MNIST);
                addLog("Loaded legacy MNIST dataset.");
            }
            else if (config.dataset == "cifar10")
            {
                dataset_loaded = dataManager->loadDataset(Dataset::CIFAR10);
                addLog("Loaded legacy CIFAR-10 dataset.");
            }
            else
            {
                addLog("Error: No suitable dataset found for this task.");
                memset(nlpInputBuffer, 0, sizeof(nlpInputBuffer));
                return;
            }
        }

        if (!dataset_loaded)
        {
            // If AI-resolved loading failed, try legacy MNIST when applicable
            std::string ds_name_lower = config.dataset_info.name;
            std::transform(ds_name_lower.begin(), ds_name_lower.end(), ds_name_lower.begin(), ::tolower);
            std::string legacy_lower = config.dataset;
            std::transform(legacy_lower.begin(), legacy_lower.end(), legacy_lower.begin(), ::tolower);
            if ((ds_name_lower.find("mnist") != std::string::npos) || (legacy_lower == "mnist"))
            {
                addLog("AI dataset load failed; falling back to built-in MNIST loader.");
                if (!dataManager->loadDataset(Dataset::MNIST))
                {
                    addLog("Error: Failed to load MNIST dataset.");
                    memset(nlpInputBuffer, 0, sizeof(nlpInputBuffer));
                    return;
                }
                dataset_loaded = true;
            }
            else
            {
                addLog("Error: Failed to load dataset.");
                memset(nlpInputBuffer, 0, sizeof(nlpInputBuffer));
                return;
            }
        }

        // Step 2: Build model architecture
        addLog("Building neural network architecture...");
        model = std::make_unique<Model>(); // Reset the model

        // Get dataset stats for shaping
        auto stats = dataManager->getDatasetStats();
        size_t dataset_input_size = stats.input_size;
        size_t dataset_num_classes = stats.num_classes;

        if (config.use_ai_architecture || config.layers.empty())
        {
            // AI-infer architecture from dataset characteristics
            addLog("Using AI-inferred architecture based on dataset characteristics.");

            // Infer architecture based on data characteristics
            std::vector<LayerConfig> inferred_layers;

            if (stats.modality == "image")
            {
                // Image classification architecture
                size_t input_size = (dataset_input_size > 0) ? dataset_input_size : 784;
                size_t num_classes = (dataset_num_classes > 0) ? dataset_num_classes : 10;

                // Create a reasonable CNN-like dense architecture for images
                inferred_layers.push_back({static_cast<int>(input_size), ActivationType::ReLU, false});

                if (input_size > 1000) // Large images (like 32x32x3 = 3072)
                {
                    inferred_layers.push_back({512, ActivationType::ReLU, false});
                    inferred_layers.push_back({256, ActivationType::ReLU, false});
                    inferred_layers.push_back({128, ActivationType::ReLU, false});
                }
                else // Smaller images (like 28x28 = 784)
                {
                    inferred_layers.push_back({256, ActivationType::ReLU, false});
                    inferred_layers.push_back({128, ActivationType::ReLU, false});
                }

                inferred_layers.push_back({static_cast<int>(num_classes), ActivationType::ReLU, true}); // Output with softmax

                addLog("Inferred image classification architecture: " + std::to_string(input_size) + " -> ... -> " + std::to_string(num_classes) + " classes");
            }
            else if (stats.modality == "tabular")
            {
                // Tabular data architecture
                size_t input_size = (dataset_input_size > 0) ? dataset_input_size : 32;
                size_t num_classes = (dataset_num_classes > 0) ? dataset_num_classes : 2;

                inferred_layers.push_back({static_cast<int>(input_size), ActivationType::ReLU, false});

                if (input_size > 100)
                {
                    inferred_layers.push_back({static_cast<int>(input_size / 2), ActivationType::ReLU, false});
                    inferred_layers.push_back({static_cast<int>(input_size / 4), ActivationType::ReLU, false});
                }
                else
                {
                    inferred_layers.push_back({64, ActivationType::ReLU, false});
                    inferred_layers.push_back({32, ActivationType::ReLU, false});
                }

                inferred_layers.push_back({static_cast<int>(num_classes), ActivationType::ReLU, true}); // Output with softmax

                addLog("Inferred tabular classification architecture: " + std::to_string(input_size) + " -> ... -> " + std::to_string(num_classes) + " classes");
            }
            else
            {
                // Generic fallback architecture
                size_t input_size = (dataset_input_size > 0) ? dataset_input_size : 128;
                size_t num_classes = (dataset_num_classes > 0) ? dataset_num_classes : 2;

                inferred_layers.push_back({static_cast<int>(input_size), ActivationType::ReLU, false});
                inferred_layers.push_back({128, ActivationType::ReLU, false});
                inferred_layers.push_back({64, ActivationType::ReLU, false});
                inferred_layers.push_back({static_cast<int>(num_classes), ActivationType::ReLU, true});

                addLog("Inferred generic architecture: " + std::to_string(input_size) + " -> 128 -> 64 -> " + std::to_string(num_classes));
            }

            config.layers = inferred_layers;
            config.is_classification = true; // Most AI tasks are classification
        }
        else
        {
            // Correct user-specified layers to match dataset
            addLog("Using user-specified architecture (aligning to dataset).");
            if (config.layers.size() >= 2)
            {
                // First layer must match input size
                config.layers.front().nodes = static_cast<int>(dataset_input_size);
                // Last layer must match number of classes
                config.layers.back().nodes = static_cast<int>(dataset_num_classes);
                config.layers.back().is_softmax = true;
                config.is_classification = true;
            }
        }

        // Build the model layers
        if (config.layers.size() < 2)
        {
            addLog("Error: Model must have at least an input and an output layer.");
            memset(nlpInputBuffer, 0, sizeof(nlpInputBuffer));
            return;
        }

        for (size_t i = 0; i < config.layers.size() - 1; i++)
        {
            const auto &current_layer_config = config.layers[i];
            const auto &next_layer_config = config.layers[i + 1];

            // Add the Dense layer connecting the current layer to the next
            addLog("Adding Dense layer: " + std::to_string(current_layer_config.nodes) + " -> " + std::to_string(next_layer_config.nodes));
            model->add(std::make_unique<Dense>(current_layer_config.nodes, next_layer_config.nodes));

            // Add the activation function for the new layer (defined by the next layer's config)
            if (next_layer_config.is_softmax)
            {
                addLog("Adding Softmax activation.");
                model->add(std::make_unique<Softmax>());
            }
            else
            {
                std::string activation_name = (next_layer_config.activation == ActivationType::ReLU) ? "ReLU" : "Sigmoid";
                addLog("Adding " + activation_name + " activation.");
                model->add(std::make_unique<Activation>(next_layer_config.activation));
            }
        }

        // Step 3: Configure optimizer and loss
        std::unique_ptr<Optimizer> opt;
        if (config.optimizer == "adam")
        {
            addLog("Using Adam optimizer.");
            auto a = std::make_unique<Adam>();
            a->setLearningRate(learningRate);
            opt = std::move(a);
        }
        else
        {
            addLog("Using SGD optimizer.");
            auto s = std::make_unique<SGD>();
            s->setLearningRate(learningRate);
            opt = std::move(s);
        }

        if (config.is_classification)
        {
            addLog("Using CrossEntropyLoss for classification.");
            model->compile(std::make_unique<CrossEntropyLoss>(), std::move(opt));
        }
        else
        {
            addLog("Using MeanSquaredError.");
            model->compile(std::make_unique<MeanSquaredError>(), std::move(opt));
        }

        addLog("AI-driven model pipeline completed successfully. Ready to train!");
    }
    else
    {
        addLog("Failed to parse command. Please try again with a valid format.");
    }

    memset(nlpInputBuffer, 0, sizeof(nlpInputBuffer));
}



void GuiManager::addLog(const std::string &message)
{
    logMessages.push_back(message);
    std::cout << "[APP_LOG] " << message << std::endl;
}
