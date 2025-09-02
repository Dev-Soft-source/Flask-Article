// Function to extract all text content from a webpage
function extractPageText() {
  // Get all text nodes in the document body
  const walker = document.createTreeWalker(
    document.body,
    NodeFilter.SHOW_TEXT,
    null,
    false
  );
  
  const textNodes = [];
  let node;
  while (node = walker.nextNode()) {
    if (node.nodeValue.trim()) {
      textNodes.push(node.nodeValue.trim());
    }
  }
  
  // Join all text nodes with newlines
  const allText = textNodes.join('\n');
  
  // Copy to clipboard (optional)
  navigator.clipboard.writeText(allText).then(() => {
    console.log('Text copied to clipboard!');
  }).catch(err => {
    console.error('Could not copy text: ', err);
  });
  
  return allText;
}

// Execute the function and log the result
console.log(extractPageText());
