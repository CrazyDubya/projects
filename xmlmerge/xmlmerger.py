#xmlmerger.py
import os

from lxml import etree


def merge_xml_files(directory, ordered_files, output_file):
    # Create a new XML root element
    root = etree.Element("root")

    # Loop through each file in the ordered list
    for file_name in ordered_files:
        file_path = os.path.join(directory, file_name)
        if os.path.exists(file_path):
            # Parse the XML file using lxml
            parser = etree.XMLParser(strip_cdata=False)
            tree = etree.parse(file_path, parser)
            root.append(tree.getroot())  # Append the root of each XML file to the new root
        else:
            print(f"Warning: {file_name} does not exist and will be skipped.")

    # Create a new ElementTree from the merged root element
    tree = etree.ElementTree(root)

    # Write the new XML to file, preserving CDATA
    output_path = os.path.join(directory, output_file)
    tree.write(output_path, pretty_print=True, xml_declaration=True, encoding='UTF-8')
    print(f"Merged XML saved to {output_path}")


# Define the directory containing the XML files
directory = '/Users/puppuccino/PycharmProjects/inner_mon/.xml'

# Define the order of the XML files to be merged
ordered_files = [
    "llmtask.xml", "challenges.xml", "creativity.xml", "knowledgeacquisition.xml",
    "reasoning.xml", "resourcesmanagement.xml", "self_monitor.xml", "timeline.xml",
    "user_interaction.xml", "prompts.xml", "prompts_mix.xml", "nextprompt.xml"
]

# Define the name of the output file
output_file = "merged_output.xml"

# Call the function to merge the XML files
merge_xml_files(directory, ordered_files, output_file)
