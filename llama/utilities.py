


def get_chunk(self, tokenizer, max_length, user_histories, strategy="truncate", max_chunk_num=None):
    # user_histories: list of user histories ids
    num_examples = 0
        
    demo_encoding_batch = []  # Stores groups of encoded exemplars
    attention_mask_batch = []
    current_group = []  # Temporary buffer for a single group
    current_mask = []
    
    for index in user_histories:
        if max_chunk_num is not None and len(demo_encoding_batch) >= max_chunk_num:
            break

        demo = self.get_user_history(index)
        demo_input_ids = tokenizer(demo).input_ids
        num_examples += 1

        if len(current_group) + len(demo_input_ids) <= max_length:
            current_group += demo_input_ids
            current_mask += [1] * len(demo_input_ids)
        else:
            # Left-padding to align groups
            pad_len = max_length - len(current_group)
            padded_group = [0] * pad_len + current_group
            padded_mask = [0] * pad_len + current_mask
            
            demo_encoding_batch.append(padded_group)
            attention_mask_batch.append(padded_mask)
            
            if len(demo_encoding_batch) < max_chunk_num:
                current_group = demo_input_ids
                current_mask = [1] * len(demo_input_ids)

    
    # Process last group if it has data
    if len(current_group) > 0:
        pad_len = max_length - len(current_group)
        padded_group = [0] * pad_len + current_group
        padded_mask = [0] * pad_len + current_mask
        demo_encoding_batch.append(padded_group)
        attention_mask_batch.append(padded_mask)
    
    return demo_encoding_batch, attention_mask_batch, num_examples
    # demo_encoding_batch: [groups, max_length], 2D list of encoded exemplars
    # attention_mask_batch: [groups, max_length], 2D list of attention masks (padding)
    # num_examples: number of examples used in the user history; we assume that we have large enough documents for each user query 
    # and we'll use several batches to process them. So we don't need to worry different users has differnt number of groups