function openTab(evt, tabName) {
  // Variables for tab content and links
  var i, tabcontent, tablinks;

  // Hide all tabcontent elements
  tabcontent = document.getElementsByClassName("tabcontent");
  for (i = 0; i < tabcontent.length; i++) {
    tabcontent[i].style.display = "none";
  }

  // Remove "active" class from all tablinks and reset input colors
  tablinks = document.getElementsByClassName("tablinks");
  for (i = 0; i < tablinks.length; i++) {
    tablinks[i].className = tablinks[i].className.replace(" active", "");
    tablinks[i].style.backgroundColor = ""; // Reset to default
  }

  // Show the current tab content and add "active" class to the clicked tab
  document.getElementById(tabName).style.display = "block";
  evt.currentTarget.className += " active";
  
  // Set the active tab color based on which tab is clicked
  if (tabName === "Terms") {
    evt.currentTarget.style.backgroundColor = "#b8b8b8";
  } else if (tabName === "Privacy") {
    evt.currentTarget.style.backgroundColor = "#d0d0d0";
  }
  
  // Optional: set a specific color for all tab content containers
  // document.querySelector('.tab-content').style.backgroundColor = evt.currentTarget.style.backgroundColor;
}
// function toggleSimilarSentences(event) {
//   const groupId = event.currentTarget.getAttribute('data-group-id');
//   const groupContainer = document.getElementById(groupId);
//   const isHidden = groupContainer.style.display === 'none';

//   groupContainer.style.display = isHidden ? 'block' : 'none'; // Toggle visibility
// }


function toggleSimilar(index) {
  var element = document.getElementById("similar-" + index);
  element.style.display = (element.style.display === "none") ? "block" : "none";
}


function scrollToOriginal(index) {
  var element = document.getElementById("original-" + index);
  if (element) {
      element.scrollIntoView({ behavior: "smooth", block: "center" });
  } else {
      console.log("Element not found: original-" + index);
  }
}

document.addEventListener('DOMContentLoaded', function () {
  const highlights = document.querySelectorAll('.highlight');

  // Create modal element
  const modal = document.createElement('div');
  modal.classList.add('modal');
  modal.innerHTML = `
      <div class="modal-content">
          <span class="close-button">&times;</span>
          <p id="definition-text"></p>
      </div>
  `;
  document.body.appendChild(modal);

  const definitionText = document.getElementById('definition-text');
  const closeButton = modal.querySelector('.close-button');

  highlights.forEach(function (highlight) {
      highlight.addEventListener('click', function (event) {
          const definition = highlight.getAttribute('data-definition');
          definitionText.textContent = definition;

          // Calculate position
          const rect = highlight.getBoundingClientRect();
          modal.style.display = 'block';

          // Position the modal above the clicked word
          modal.style.left = `${rect.left + window.scrollX}px`;
          modal.style.top = `${rect.top + window.scrollY - modal.offsetHeight - 10}px`;

          event.stopPropagation();
      });
  });

  closeButton.addEventListener('click', function () {
      modal.style.display = 'none';
  });

  window.addEventListener('click', function (event) {
      if (event.target === modal) {
          modal.style.display = 'none';
      }
  });
});
// function processText(formId) {
//   const formData = new FormData(document.getElementById(formId));
//   fetch('/process', {
//       method: 'POST',
//       body: formData
//   })
//   .then(response => response.json())
//   .then(data => {
//       displayFlaggedSentences(data);
//   })
//   .catch(error => {
//       console.error('Error:', error);
//   });
// }

function displayFlaggedSentences(flaggedSentences) {
  const resultsContainer = document.getElementById('results');
  resultsContainer.innerHTML = ''; // Clear previous results

  if (flaggedSentences.length > 0) {
    const list = document.createElement('ul');
    
    flaggedSentences.forEach((group, index) => {
      const groupContainer = document.createElement('li');
      
      // Create and append the main sentence.
      const mainSentence = document.createElement('strong');
      mainSentence.textContent = group.mainSentence[0]; // Assuming mainSentence is a tuple
      groupContainer.appendChild(mainSentence);

      // Create a container for the similar sentences to always display them.
      const similarContainer = document.createElement('div');
      similarContainer.id = `group-${index}`;
      
      const similarList = document.createElement('ul');
      group.similar_sentences.forEach(sentence => {
        const item = document.createElement('li');
        item.textContent = sentence[0]; // Assuming each sentence in similar_sentences is a tuple
        similarList.appendChild(item);
      });
      
      similarContainer.appendChild(similarList);
      groupContainer.appendChild(similarContainer);
      list.appendChild(groupContainer);
    });

    resultsContainer.appendChild(list);
  } else {
    resultsContainer.textContent = 'No flagged sentences found.';
  }
}

document.addEventListener("DOMContentLoaded", function() {
  console.log("Script loaded"); // Add this line to verify

  window.toggleSimilar = function(index) {
    var elem = document.getElementById("similar-" + index);
    elem.style.display = elem.style.display === "none" ? "block" : "none";
  };

  window.showContext = function(index, sentences) {
    var contextPopUp = document.getElementById("context-popup");
    var contextText = document.getElementById("context-text");
    
    // Clear the previous content
    contextText.innerHTML = '';

    // Create a single paragraph element to contain all sentences
    var paragraphElement = document.createElement("div");
    paragraphElement.style.whiteSpace = "pre-wrap"; // Preserve whitespace and newlines

    // Iterate over the list of sentences
    for (var i = 0; i < sentences.length; i++) {
      var sentence = sentences[i];
      
      // Highlight the flagged sentence in red
      if (i == index) {
        var highlightedSentenceElement = document.createElement("span");
        highlightedSentenceElement.style.color = "rgba(30, 10, 10, 1)";
        highlightedSentenceElement.style.fontWeight = "bold";
        highlightedSentenceElement.textContent = sentence;
        paragraphElement.appendChild(highlightedSentenceElement);
      } else {
        // Append other sentences normally
        var sentenceElement = document.createElement("span");
        sentenceElement.style.color = "rgba(72, 72, 72, 0.65)";
        sentenceElement.textContent = sentence;
        paragraphElement.appendChild(sentenceElement);
      }

      // Add a space or newline after each sentence
      paragraphElement.appendChild(document.createTextNode('\n'));
    }

    // Append the paragraph element to the context text
    contextText.appendChild(paragraphElement);

    // Display the pop-up window
    contextPopUp.style.display = "flex";

    // Ensure the highlighted sentence is visible immediately
    if (highlightedSentenceElement) {
      var offsetTop = highlightedSentenceElement.offsetTop;
      var contextTextHeight = contextText.clientHeight;
      var sentenceHeight = highlightedSentenceElement.clientHeight;
      var scrollPosition = offsetTop - (contextTextHeight / 2) + (sentenceHeight / 2);
      contextText.scrollTop = scrollPosition;
    }
  };

  window.closeContext = function() {
    var contextPopUp = document.getElementById("context-popup");
    contextPopUp.style.display = "none";
  };
});

document.addEventListener('DOMContentLoaded', function () {
  const highlights = document.querySelectorAll('.highlight');
  const closeButton = modal.querySelector('.close-button');
  // Create modal element
  const modal = document.createElement('div');
  modal.classList.add('modal');
  modal.innerHTML = `
      <div class="modal-content">
          <span class="close-button">&times;</span>
          <p id="definition-text"></p>
      </div>
  `;
  document.body.appendChild(modal);

  const definitionText = document.getElementById('definition-text');
 

  // Set the margin distance between the modal and the word
  const margin = 10; // Adjust this value to control distance

  highlights.forEach(function (highlight) {
      highlight.addEventListener('click', function (event) {
          const definition = highlight.getAttribute('data-definition');
          definitionText.textContent = definition;

          // Calculate position
          const rect = highlight.getBoundingClientRect();
          modal.style.display = 'block';

          // Position the modal above the clicked word with the specified margin
          const left = rect.left + window.scrollX;
          const top = rect.top + window.scrollY - modal.offsetHeight - margin;

          // Ensure the modal is not out of the viewport bounds
          modal.style.left = `${Math.max(left, 0)}px`;
          modal.style.top = `${Math.max(top, 0)}px`;

          event.stopPropagation();
      });
  });

  closeButton.addEventListener('click', function (event) {
      modal.style.display = 'none';
      event.stopPropagation();
  });

  // Add event listener to the document to handle clicks outside the modal
  document.addEventListener('click', function (event) {
      if (!modal.contains(event.target) && !event.target.classList.contains('highlight')) {
          modal.style.display = 'none';
      }
  });
});

function showLoading() {
  document.getElementById('loading').style.display = 'flex';

  // Delay the appearance of the text message
  setTimeout(function() {
      document.getElementById('loadingText').style.opacity = 1;
      document.getElementById('loadingImage').style.display = 'block';
  }, 3000);

  setTimeout(function() {
    document.getElementById('loadingText2').style.opacity = 1;
  }, 5000);

}

// Ensure loading screen is hidden on page load
window.addEventListener('pageshow', function(event) {
  if (event.persisted || (window.performance && window.performance.navigation.type == 2)) {
      document.getElementById('loading').style.display = 'none';
  }
});

window.onload = function() {
  document.getElementById('loading').style.display = 'none';
};