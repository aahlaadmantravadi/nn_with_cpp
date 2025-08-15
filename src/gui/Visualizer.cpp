#include "gui/Visualizer.h"



#include "imgui.h"
#include "nn/Model.h"
#include "nn/layers/Dense.h"



#include <algorithm>
#include <string>
#include <vector>



Visualizer::Visualizer()
{
}



void Visualizer::render(Model *model)
{
    if ((!model) || model->getLayers().empty())
    {
        ImGui::Text("No model to visualize.");
        return;
    }

    // Get the available space in the window for drawing
    ImVec2 available_size = ImGui::GetContentRegionAvail();
    ImVec2 p = ImGui::GetCursorScreenPos();
    ImDrawList *draw_list = ImGui::GetWindowDrawList();

    // Calculate the canvas area
    float canvas_width = available_size.x * 0.95f; // Use 95% of available width
    float canvas_height = available_size.y * 0.9f; // Use 90% of available height

    // Draw a frame around our visualization area
    ImVec2 canvas_p0 = ImVec2(p.x, p.y);
    ImVec2 canvas_p1 = ImVec2(p.x + canvas_width, p.y + canvas_height);
    draw_list->AddRect(canvas_p0, canvas_p1, IM_COL32(60, 60, 60, 255));

    // Count the layers and determine input/output dimensions
    size_t num_dense_layers = 0;
    size_t input_size = 0;
    size_t output_size = 0;

    // First pass to count layers and get input/output sizes
    for (const auto &layer : model->getLayers())
    {
        if (auto *dense_layer = dynamic_cast<Dense *>(layer.get()))
        {
            if (num_dense_layers == 0)
            {
                // First dense layer gives us input size from weights
                input_size = dense_layer->weights.getRows();
            }
            output_size = dense_layer->biases.getCols(); // Last one will be final output size
            num_dense_layers++;
        }
    }

    if (num_dense_layers == 0)
    {
        ImGui::Text("No dense layers to visualize.");
        return;
    }

    // Add 2 for input and output layers
    size_t total_layers = num_dense_layers + 1;

    // Calculate horizontal spacing between layers
    float x_spacing = canvas_width / (total_layers + 1);
    float x = p.x + x_spacing; // Start position
    float y_center = p.y + (canvas_height / 2.0f); // Vertical center of canvas

    std::vector<ImVec2> prev_layer_nodes;
    size_t layer_idx = 0;

    // Render input layer first
    std::vector<ImVec2> input_layer_nodes;
    size_t input_nodes = input_size;

    // Limit the number of visible nodes if there are too many
    const size_t max_visible_nodes = 20;
    size_t visible_input_nodes = std::min(input_nodes, max_visible_nodes);

    // Calculate node spacing and size based on available space
    float max_node_spacing = 30.0f * ImGui::GetIO().FontGlobalScale;
    float node_spacing = std::min(max_node_spacing, canvas_height / (visible_input_nodes + 1));
    float node_radius = std::min(5.0f * ImGui::GetIO().FontGlobalScale, node_spacing / 3.0f);

    // Calculate total height needed for this layer
    float layer_height = node_spacing * (visible_input_nodes - 1);

    // Center the layer vertically
    float y_start = y_center - (layer_height / 2.0f);

    // Create nodes for input layer
    for (size_t i = 0; i < visible_input_nodes; i++)
    {
        ImVec2 node_pos(x, y_start + i * node_spacing);
        input_layer_nodes.push_back(node_pos);
    }

    // If there are more nodes than we're showing, add ellipsis
    if (input_nodes > visible_input_nodes)
    {
        float ellipsis_y = y_start + visible_input_nodes * node_spacing + node_spacing;
        ImDrawList *draw_list_ptr = ImGui::GetWindowDrawList(); // Use a local pointer
        draw_list_ptr->AddText(ImVec2(x - 10, ellipsis_y), IM_COL32(200, 200, 200, 255), "...");
    }

    // Draw the input nodes with green color
    for (const auto &node_pos : input_layer_nodes)
    {
        draw_list->AddCircleFilled(node_pos, node_radius, IM_COL32(120, 200, 120, 255)); // Green for input
    }

    // Add layer size as text below
    std::string input_layer_info = std::to_string(input_nodes);
    draw_list->AddText(
        ImVec2(x - ImGui::CalcTextSize(input_layer_info.c_str()).x / 2,
               y_center + layer_height / 2 + 15),
        IM_COL32(120, 200, 120, 255),
        input_layer_info.c_str()
    );

    prev_layer_nodes = input_layer_nodes;
    x += x_spacing;

    // Draw the network
    for (const auto &layer : model->getLayers())
    {
        if (auto *dense_layer = dynamic_cast<Dense *>(layer.get()))
        {
            std::vector<ImVec2> current_layer_nodes;

            // Get number of nodes in this layer
            size_t num_nodes = dense_layer->biases.getCols();

            // Limit the number of visible nodes if there are too many
            const size_t max_visible_nodes_inner = 20;
            size_t visible_nodes = std::min(num_nodes, max_visible_nodes_inner);

            // Calculate node spacing and size based on available space
            float max_node_spacing_inner = 30.0f * ImGui::GetIO().FontGlobalScale;
            float node_spacing_inner = std::min(max_node_spacing_inner, canvas_height / (visible_nodes + 1));
            float node_radius_inner = std::min(5.0f * ImGui::GetIO().FontGlobalScale, node_spacing_inner / 3.0f);

            // Calculate total height needed for this layer
            float layer_height_inner = node_spacing_inner * (visible_nodes - 1);

            // Center the layer vertically
            float y_start_inner = y_center - (layer_height_inner / 2.0f);

            // Create nodes for this layer
            for (size_t i = 0; i < visible_nodes; i++)
            {
                ImVec2 node_pos(x, y_start_inner + i * node_spacing_inner);
                current_layer_nodes.push_back(node_pos);
            }

            // If there are more nodes than we're showing, add ellipsis
            if (num_nodes > visible_nodes)
            {
                float ellipsis_y = y_start_inner + visible_nodes * node_spacing_inner + node_spacing_inner;
                ImDrawList *draw_list_ptr = ImGui::GetWindowDrawList(); // Use a local pointer
                draw_list_ptr->AddText(ImVec2(x - 10, ellipsis_y), IM_COL32(200, 200, 200, 255), "...");
            }

            // Draw connections from previous layer
            if (!prev_layer_nodes.empty())
            {
                for (const auto &prev_node : prev_layer_nodes)
                {
                    for (const auto &current_node : current_layer_nodes)
                    {
                        // Draw connection with transparency
                        draw_list->AddLine(prev_node, current_node, IM_COL32(200, 200, 200, 40));
                    }
                }
            }

            // Draw the nodes
            for (const auto &node_pos : current_layer_nodes)
            {
                draw_list->AddCircleFilled(node_pos, node_radius_inner, IM_COL32(255, 255, 255, 255));
            }

            // Add layer size as text below
            std::string layer_info = std::to_string(num_nodes);
            draw_list->AddText(
                ImVec2(x - ImGui::CalcTextSize(layer_info.c_str()).x / 2,
                       y_center + layer_height_inner / 2 + 15),
                IM_COL32(200, 200, 200, 255),
                layer_info.c_str()
            );

            // Store this layer's nodes for the next iteration
            prev_layer_nodes = current_layer_nodes;
            x += x_spacing;
            layer_idx++;
        }
    }

    // Now render the output layer if we have a final output size
    if (output_size > 0 && (!prev_layer_nodes.empty()))
    {
        std::vector<ImVec2> output_layer_nodes;
        size_t visible_output_nodes = std::min(output_size, max_visible_nodes);

        // Calculate node spacing and size for output layer
        float output_node_spacing = std::min(max_node_spacing, canvas_height / (visible_output_nodes + 1));
        float output_layer_height = output_node_spacing * (visible_output_nodes - 1);
        float output_y_start = y_center - (output_layer_height / 2.0f);

        // Create nodes for output layer
        for (size_t i = 0; i < visible_output_nodes; i++)
        {
            ImVec2 node_pos(x, output_y_start + i * output_node_spacing);
            output_layer_nodes.push_back(node_pos);
        }

        // If there are more nodes than we're showing, add ellipsis
        if (output_size > visible_output_nodes)
        {
            float ellipsis_y = output_y_start + visible_output_nodes * output_node_spacing + output_node_spacing;
            ImDrawList *draw_list_ptr = ImGui::GetWindowDrawList(); // Use a local pointer
            draw_list_ptr->AddText(ImVec2(x - 10, ellipsis_y), IM_COL32(200, 200, 200, 255), "...");
        }

        // Draw connections from previous layer to output
        for (const auto &prev_node : prev_layer_nodes)
        {
            for (const auto &output_node : output_layer_nodes)
            {
                draw_list->AddLine(prev_node, output_node, IM_COL32(200, 200, 200, 40));
            }
        }

        // Draw the output nodes
        for (const auto &node_pos : output_layer_nodes)
        {
            draw_list->AddCircleFilled(node_pos, node_radius, IM_COL32(200, 120, 120, 255)); // Red for output
        }

        // Add layer size as text below
        std::string output_layer_info = std::to_string(output_size);
        draw_list->AddText(
            ImVec2(x - ImGui::CalcTextSize(output_layer_info.c_str()).x / 2,
                   y_center + output_layer_height / 2 + 15),
            IM_COL32(200, 120, 120, 255),
            output_layer_info.c_str()
        );
    }

    // Add some space after the visualization
    ImGui::Dummy(ImVec2(0, canvas_height + 20));
}
