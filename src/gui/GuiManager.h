// =============================================================================
// File: src/gui/GuiManager.h
// =============================================================================
//
// Description: Declares the GuiManager class, which is the heart of the
//              application. It manages the main window (using GLFW), the GUI
//              rendering (using ImGui), and orchestrates the interactions
//              between the user, the neural network, and the data manager.
//
// =============================================================================

#pragma once



#include "imgui.h" // Include ImGui for ImVec2
#include "nn/nn_types.h" // For Backend enum



// Forward declarations to avoid including heavy headers
struct GLFWwindow;
class Model;
class DataManager;
class Parser;
class Visualizer;



#include <memory>
#include <string>
#include <vector>



class GuiManager
{
public:
    // Constructor: Initializes member variables.
    GuiManager();

    // Destructor: Handles cleanup of resources.
    ~GuiManager();

    // The main public method to start the entire application.
    void run();

private:
    // --- Initialization and Shutdown ---
    void init();
    void shutdown();

    // --- Main Application Loop ---
    void mainLoop();

    // --- UI Rendering Methods ---
    void renderUI();
    void renderMenuBar();
    void renderControlPanel();
    void renderLogPanel();
    void renderVisualizationWindow();

    // Add a log entry to both the on-screen log and stdout
    void addLog(const std::string &message);

    // --- Helper Methods ---
    void processNlpInput();
    void renderDragHandle(const char *id);

    // --- Member Variables ---

    // Windowing and GUI
    GLFWwindow *window;
    int windowWidth;
    int windowHeight;

    // UI State
    char nlpInputBuffer[1024];
    char assistantInputBuffer[1024];
    std::vector<std::string> logMessages;
    Backend selectedBackend;
    bool showVisualizerWindow;
    float uiScale;
    int numEpochs = 10;  // Default to 10 epochs

    // Training metrics
    float currentLoss = 0.0f;
    float testAccuracy = 0.0f;
    float testLoss = 0.0f;
    bool showTestResults = false;
    // Training progress
    int currentEpoch = 0;
    size_t batchSize = 64;
    size_t numBatchesPerEpoch = 0;
    size_t currentBatchIndex = 0;
    float learningRate = 0.001f;
    bool debugVerbose = false;

    // Core Application Components (using smart pointers for automatic memory management)
    std::unique_ptr<Model> model;
    std::unique_ptr<DataManager> dataManager;
    std::unique_ptr<Parser> nlpParser;
    std::unique_ptr<Visualizer> visualizer;

    // System capabilities
    bool hasCuda = false;
};
