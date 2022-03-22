# Python program for insert and search
# operation in a Trie

class TrieNode:
	
	# Trie node class
	def __init__(self):
		self.children = [None]*26

		# isEndOfWord is True if node represent the end of the word
		self.last = False

		self.loc = -1 # to store location of word in outout.csv / lists

		self.ch = None

class Trie:
	
	# Trie data structure class
	def __init__(self):
		self.root = self.getNode()

	def getNode(self):
	
		# Returns new trie node (initialized to NULLs)
		return TrieNode()

	def _charToIndex(self,ch):
		
		# private helper function
		# Converts key current character into index
		# use only 'a' through 'z' and lower case
		
		return ord(ch)-ord('a')


	def insert(self,key, x):
		
		# If not present, inserts key into trie
		# If the key is prefix of trie node,
		# just marks leaf node
		pCrawl = self.root
		length = len(key)
		for level in range(length):
			index = self._charToIndex(key[level])

			# if current character is not present
			if not pCrawl.children[index]:
				pCrawl.children[index] = self.getNode()
			pCrawl = pCrawl.children[index]

		# mark last node as leaf
		pCrawl.last = True
		pCrawl.loc = x							#index of word added as 'loc'
		pCrawl.ch = key[level]

	def search(self, key):
		
		# Search key in the trie
		# Returns true if key presents
		# in trie, else false
		pCrawl = self.root
		length = len(key)
		for level in range(length):
			index = self._charToIndex(key[level])
			if not pCrawl.children[index]:
				return False
			pCrawl = pCrawl.children[index]

		# return pCrawl.isEndOfWord
		return pCrawl.loc								#returns the index of given word in the array/output.csv


	# for autocomplete :::
	
	def suggestionsRec(self, node, word):
		
		# Method to recursively traverse the trie
        # and return a whole word.
		if node.last:
			print(word)
 
		# for a, n in node.children.items():
		# 	self.suggestionsRec(n, word + a)

		# for n in node.children:
		# 	self.suggestionsRec(n, word+)
		
 
	def printAutoSuggestions(self, key):
 
        # Returns all the words in the trie whose common
        # prefix is the given key thus listing out all
        # the suggestions for autocomplete.
		node = self.root
 
		for a in key:
            # no string in the Trie has this prefix
			# if not node.children.get(a):
			index = self._charToIndex(a)
			if node.children[self._charToIndex(a)]==[None]:
				return 0
			node = node.children[self._charToIndex(a)]
 
        # If prefix is present as a word, but
        # there is no subtree below the last
        # matching node.
		if not node.children:
			return -1
 
		self.suggestionsRec(node, key)
		return 1