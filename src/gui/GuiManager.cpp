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
#include "nn/Model.h"
#include "data/DataManager.h"
#include "nlp/Parser.h"
#include "gui/Visualizer.h"
#include "nn/layers/Dense.h"
#include "nn/layers/Activation.h"
#include "nn/layers/Softmax.h"
#include "nn/Loss.h"
#include "nn/optimizers/SGD.h"
#include "nn/optimizers/Adam.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>

#include <iostream>
#include <stdexcept>
#include <thread>
#include <atomic>
#include <string>
#include <future>

// --- Global state for training thread ---
std::atomic<bool> isTraining(false);
std::thread trainingThread;

// GLFW error callback function
static void glfw_error_callback(int error, const char* description) {
    std::cerr << "GLFW Error " << error << ": " << description << std::endl;
}

// Helper function to render selectable, multi-line wrapped text
void RenderSelectableWrappedText(const char* label, const std::string& text) {
    ImGui::PushStyleColor(ImGuiCol_FrameBg, IM_COL32(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_Border, IM_COL32(0, 0, 0, 0));
    ImGui::InputTextMultiline(label, (char*)text.c_str(), text.length() + 1, ImVec2(-1.0f, ImGui::GetTextLineHeight() * 4), ImGuiInputTextFlags_ReadOnly);
    ImGui::PopStyleColor(2);
}


// --- GuiManager Implementation ---

GuiManager::GuiManager()
    : window(nullptr), windowWidth(1600), windowHeight(900),
      selectedBackend(Backend::CPU), showVisualizerWindow(true), uiScale(2.0f) { // Default scale to 2.0x
    memset(nlpInputBuffer, 0, sizeof(nlpInputBuffer));
    logMessages.push_back("Welcome to TensorFlow from Scratch!");
}

GuiManager::~GuiManager() {
    isTraining = false;
    if (trainingThread.joinable()) {
        trainingThread.join();
    }
}

void GuiManager::run() {
    try {
        init();
        mainLoop();
        shutdown();
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed in GuiManager::run(): ") + e.what());
    }
}

void GuiManager::init() {
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit()) throw std::runtime_error("Failed to initialize GLFW");

    const char* glsl_version = "#version 330";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    window = glfwCreateWindow(windowWidth, windowHeight, "TensorFlow from Scratch", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        throw std::runtime_error("Failed to create GLFW window");
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui::StyleColorsDark();
    
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    model = std::make_unique<Model>();
    dataManager = std::make_unique<DataManager>();
    nlpParser = std::make_unique<Parser>();
    visualizer = std::make_unique<Visualizer>();
    
    logMessages.push_back("GUI Manager initialized. Enter a command to begin.");
}

void GuiManager::shutdown() {
    isTraining = false;
    if (trainingThread.joinable()) {
        trainingThread.join();
    }
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    std::cout << "Application shut down gracefully." << std::endl;
}

void GuiManager::mainLoop() {
    while (!glfwWindowShouldClose(window)) {
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

void GuiManager::renderUI() {
    ImGui::GetIO().FontGlobalScale = uiScale;

    renderMenuBar();
    renderControlPanel();
    renderLogPanel();
    if (showVisualizerWindow) {
        renderVisualizationWindow();
    }
}

void GuiManager::renderMenuBar() {
    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("Exit")) { glfwSetWindowShouldClose(window, true); }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("View")) {
            ImGui::MenuItem("Visualizer", NULL, &showVisualizerWindow);
            ImGui::Separator();
            ImGui::Text("UI Scale");
            ImGui::SliderFloat("##ui_scale", &uiScale, 0.5f, 2.5f);
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }
}

void GuiManager::renderControlPanel() {
    ImGui::Begin("Control Panel");
    ImGui::Text("Natural Language Network Configuration");
    ImGui::Separator();

    if (ImGui::InputText("Command", nlpInputBuffer, sizeof(nlpInputBuffer), ImGuiInputTextFlags_EnterReturnsTrue)) {
        processNlpInput();
    }
    ImGui::SameLine();
    if (ImGui::Button("Parse & Build")) {
        processNlpInput();
    }
    RenderSelectableWrappedText("##eg1", "e.g., 'build 784-128-relu-64-relu-10-softmax with adam for mnist'");
    RenderSelectableWrappedText("##eg2", "e.g., 'train a network to classify handwritten digits'");

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Text("Training Controls");

    if (ImGui::RadioButton("CPU", selectedBackend == Backend::CPU)) { selectedBackend = Backend::CPU; }
    ImGui::SameLine();
    if (ImGui::RadioButton("GPU (CUDA)", selectedBackend == Backend::GPU)) { selectedBackend = Backend::GPU; }

    ImGui::Spacing();

    // Add epoch input control
    ImGui::Text("Epochs:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(60);
    ImGui::InputInt("##epochs", &numEpochs, 1, 5);
    
    // Clamp to reasonable values
    if (numEpochs < 1) numEpochs = 1;
    if (numEpochs > 100) numEpochs = 100;

    ImGui::Spacing();
    
    // Make buttons wider as requested by user
    float buttonWidth = 240;
    
    if (ImGui::Button("Start Training", ImVec2(buttonWidth, 30))) {
        if (!isTraining) {
            // Reset training metrics for new session
            currentLoss = 0.0f;
            showTestResults = false;
            isTraining = true;
            
            // Set the selected backend before starting training
            try {
                if (selectedBackend == Backend::GPU) {
                    logMessages.push_back("Using GPU backend for training...");
                    model->setBackend(Backend::GPU);
                } else {
                    logMessages.push_back("Using CPU backend for training...");
                    model->setBackend(Backend::CPU);
                }
                
                // Create stable shared copies of data for the thread
                auto model_ptr = model.get();
                auto data_ptr = dataManager.get();
                auto log_ptr = &logMessages;
                auto epochs_to_use = numEpochs; // Capture epochs from UI
                auto current_loss_ptr = &currentLoss;
                
                // Detach any previous thread
                if (trainingThread.joinable()) {
                    trainingThread.detach();
                }
                
                // Create new thread with captured pointers rather than 'this'
                trainingThread = std::thread([model_ptr, data_ptr, log_ptr, epochs_to_use, current_loss_ptr]() {
                    try {
                        log_ptr->push_back("Training started...");
                        size_t batch_size = 64;
                        size_t num_batches = data_ptr->getTrainSamplesCount() / batch_size;
                        
                        for(size_t i = 0; i < epochs_to_use && isTraining; ++i) {
                            float epoch_loss = 0;
                            for(size_t j = 0; j < num_batches && isTraining; ++j) {
                                try {
                                    auto batch = data_ptr->getTrainBatch(batch_size);
                                    epoch_loss += model_ptr->train_step(batch.first, batch.second);
                                } catch (const std::exception& e) {
                                    log_ptr->push_back("Error in training batch: " + std::string(e.what()));
                                    // Continue with next batch
                                }
                            }
                            
                            float avg_loss = epoch_loss / num_batches;
                            *current_loss_ptr = avg_loss; // Update the loss for UI
                            log_ptr->push_back("Epoch " + std::to_string(i+1) + " Loss: " + std::to_string(avg_loss));
                        }
                        
                        isTraining = false;
                        log_ptr->push_back("Training finished.");
                    } catch (const std::exception& e) {
                        log_ptr->push_back("Training error: " + std::string(e.what()));
                        isTraining = false;
                    }
                });
                
                // Detach the thread to prevent crashes
                trainingThread.detach();
            } catch (const std::exception& e) {
                logMessages.push_back("Error starting training: " + std::string(e.what()));
                isTraining = false;
            }
        }
    }
    
    ImGui::Spacing();
    
    if (ImGui::Button("Stop Training", ImVec2(buttonWidth, 30))) {
        if (isTraining) {
            logMessages.push_back("Stopping training...");
            isTraining = false;
            
            // Wait for training thread to finish with a timeout
            auto join_result = std::future<void>();
            if (trainingThread.joinable()) {
                // Use a separate thread to join with a timeout to avoid deadlock
                std::thread([&trainingThread = trainingThread]() {
                    if (trainingThread.joinable()) {
                        trainingThread.join();
                    }
                }).detach();
            }
            
            logMessages.push_back("Training stopped.");
        }
    }
    
    ImGui::Spacing();
    
    if (ImGui::Button("Test Model", ImVec2(buttonWidth, 30))) {
        if (!isTraining && model && dataManager) {
            try {
                logMessages.push_back("Testing model on test data...");
                
                // Get the test data
                auto X_test = dataManager->getTestData();
                auto y_test = dataManager->getTestLabels();
                
                // Evaluate the model
                auto [loss, accuracy] = model->evaluate(X_test, y_test);
                
                // Update metrics
                testLoss = loss;
                testAccuracy = accuracy;
                showTestResults = true;
                
                logMessages.push_back("Test Loss: " + std::to_string(testLoss));
                logMessages.push_back("Test Accuracy: " + std::to_string(testAccuracy * 100.0f) + "%");
            } catch (const std::exception& e) {
                logMessages.push_back("Error testing model: " + std::string(e.what()));
            }
        } else if (isTraining) {
            logMessages.push_back("Cannot test while training is in progress. Stop training first.");
        } else {
            logMessages.push_back("No model or data available for testing.");
        }
    }
    
    // Display metrics
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Text("Metrics");
    
    // Show current training loss if available
    if (currentLoss > 0.0f) {
        ImGui::Text("Current Training Loss: %.6f", currentLoss);
    }
    
    // Show test results if available
    if (showTestResults) {
        ImGui::Text("Test Loss: %.6f", testLoss);
        ImGui::Text("Test Accuracy: %.2f%%", testAccuracy * 100.0f);
    }
    
    // Add some space
    ImGui::Dummy(ImVec2(0, 15));
    
    // Add bottom drag handle
    renderDragHandle("ctrl");
    ImGui::End();
}

void GuiManager::renderLogPanel() {
    ImGui::Begin("Log");
    
    // Main content
    for (size_t i = 0; i < logMessages.size(); ++i) {
        std::string label = "##log" + std::to_string(i);
        RenderSelectableWrappedText(label.c_str(), logMessages[i]);
    }
    if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY()) {
        ImGui::SetScrollHereY(1.0f);
    }
    
    // Add some padding
    ImGui::Dummy(ImVec2(0, 10));
    
    // Add bottom drag handle
    renderDragHandle("log");
    ImGui::End();
}

void GuiManager::renderVisualizationWindow() {
    ImGui::Begin("Network Visualizer", &showVisualizerWindow);
    ImGui::Text("Live Training Visualization");
    ImGui::Separator();
    if (visualizer) {
        visualizer->render(model.get());
    }
    
    // Add some padding
    ImGui::Dummy(ImVec2(0, 10));
    
    // Add bottom drag handle
    renderDragHandle("vis");
    ImGui::End();
}

void GuiManager::renderDragHandle(const char* id) {
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
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    float dot_spacing = 8.0f * uiScale;
    
    // Center the dots horizontally
    float start_x = button_pos.x + (button_size.x / 2.0f) - (2.0f * dot_spacing);
    float y = button_pos.y + (button_size.y / 2.0f);
    
    for (int i = 0; i < 5; i++) {
        draw_list->AddCircleFilled(
            ImVec2(start_x + (i * dot_spacing), y),
            1.5f * uiScale,
            IM_COL32(200, 200, 200, 180)
        );
    }
    
    ImGui::PopStyleVar();
    
    // Make the entire window draggable when interacting with the handle
    if (ImGui::IsItemActive() && ImGui::IsMouseDragging(0)) {
        ImGui::SetWindowFocus();
        ImVec2 windowPos = ImGui::GetWindowPos();
        ImVec2 delta = ImGui::GetIO().MouseDelta;
        ImGui::SetWindowPos(ImVec2(windowPos.x + delta.x, windowPos.y + delta.y));
    }
    
    ImGui::PopStyleColor(3);
}

void GuiManager::processNlpInput() {
    if (strlen(nlpInputBuffer) == 0) return;

    std::string command(nlpInputBuffer);
    logMessages.push_back("Parsing command: " + command);
    
    ModelConfig config = nlpParser->parse(command);
    
    if (config.valid) {
        logMessages.push_back("Building model...");
        model = std::make_unique<Model>(); // Reset the model
        
        // --- Corrected Model Building Logic ---
        if (config.layers.size() < 2) {
            logMessages.push_back("Error: Model must have at least an input and an output layer.");
            return;
        }

        for (size_t i = 0; i < config.layers.size() - 1; ++i) {
            const auto& current_layer_config = config.layers[i];
            const auto& next_layer_config = config.layers[i+1];

            // Add the Dense layer connecting the current layer to the next
            logMessages.push_back("Adding Dense layer: " + std::to_string(current_layer_config.nodes) + " -> " + std::to_string(next_layer_config.nodes));
            model->add(std::make_unique<Dense>(current_layer_config.nodes, next_layer_config.nodes));

            // Add the activation function for the new layer (defined by the next layer's config)
            if (next_layer_config.is_softmax) {
                logMessages.push_back("Adding Softmax activation.");
                model->add(std::make_unique<Softmax>());
            } else {
                std::string activation_name = (next_layer_config.activation == ActivationType::ReLU) ? "ReLU" : "Sigmoid";
                logMessages.push_back("Adding " + activation_name + " activation.");
                model->add(std::make_unique<Activation>(next_layer_config.activation));
            }
        }

        std::unique_ptr<Optimizer> opt;
        if (config.optimizer == "adam") {
            logMessages.push_back("Using Adam optimizer.");
            opt = std::make_unique<Adam>();
        } else {
            logMessages.push_back("Using SGD optimizer.");
            opt = std::make_unique<SGD>();
        }

        if(config.is_classification){
            logMessages.push_back("Using CrossEntropyLoss for classification.");
            model->compile(std::make_unique<CrossEntropyLoss>(), std::move(opt));
        } else {
            logMessages.push_back("Using MeanSquaredError.");
            model->compile(std::make_unique<MeanSquaredError>(), std::move(opt));
        }
        
        logMessages.push_back("Model built successfully. Loading data...");
        if (config.dataset == "mnist") {
            if (dataManager->loadDataset(Dataset::MNIST)) {
                logMessages.push_back("MNIST data loaded. Ready to train.");
            } else {
                logMessages.push_back("Error: Failed to load MNIST data.");
            }
        } else {
            logMessages.push_back("Warning: Dataset '" + config.dataset + "' is not supported yet.");
        }

    } else {
        logMessages.push_back("Failed to parse command. Please try again with a valid format.");
    }
    
    memset(nlpInputBuffer, 0, sizeof(nlpInputBuffer));
}
